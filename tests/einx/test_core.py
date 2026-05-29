"""Layer-0 tests for the xrtoolz.einx core verbs."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import xrtoolz.einx as xnx
from xrtoolz.einx import CoordMismatch, PatternError


@pytest.fixture
def field() -> xr.DataArray:
    return xr.DataArray(
        np.arange(18.0).reshape(3, 2, 3),
        dims=("time", "lat", "lon"),
        coords={"time": np.arange(3), "lat": [0.0, 1.0], "lon": [0.0, 1.0, 2.0]},
        name="ssh",
    )


@pytest.fixture
def mask() -> xr.DataArray:
    return xr.DataArray(
        np.ones((2, 3)),
        dims=("lat", "lon"),
        coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0, 2.0]},
    )


def test_einsum_contraction_matches_xarray(field, mask) -> None:
    out = xnx.einsum("time lat lon, lat lon -> time", field, mask)
    assert out.dims == ("time",)
    np.testing.assert_allclose(out.values, field.sum(("lat", "lon")).values)
    np.testing.assert_array_equal(
        out.coords["time"].values, field.coords["time"].values
    )


def test_einsum_is_transpose_invariant(field, mask) -> None:
    """Pattern is by dim name, so input dim order must not matter."""
    permuted = field.transpose("lon", "time", "lat")
    out = xnx.einsum("time lat lon, lat lon -> time", permuted, mask)
    np.testing.assert_allclose(out.values, field.sum(("lat", "lon")).values)


def test_einsum_rejects_dim_not_on_input(field, mask) -> None:
    with pytest.raises(PatternError, match="does not match"):
        xnx.einsum("time x lon, lat lon -> time", field, mask)


def test_einsum_wrong_input_count(field, mask) -> None:
    with pytest.raises(PatternError, match="declares 2 input"):
        xnx.einsum("time lat lon, lat lon -> time", field)


def test_einsum_coord_mismatch_raises(field, mask) -> None:
    shifted = mask.assign_coords(lon=mask.coords["lon"] + 100)
    with pytest.raises(CoordMismatch, match="lon"):
        xnx.einsum("time lat lon, lat lon -> time", field, shifted)


def test_einsum_align_inner_joins(field) -> None:
    partial = xr.DataArray(
        np.ones((2, 2)),
        dims=("lat", "lon"),
        coords={"lat": [0.0, 1.0], "lon": [1.0, 2.0]},
    )
    out = xnx.einsum("time lat lon, lat lon -> time", field, partial, align=True)
    # Inner join keeps lon in {1, 2}; compare against the same subset.
    expected = field.sel(lon=[1.0, 2.0]).sum(("lat", "lon"))
    np.testing.assert_allclose(out.values, expected.values)


def test_reduce_native_op(field) -> None:
    out = xnx.reduce("time lat lon -> lat lon", field, op="mean")
    assert out.dims == ("lat", "lon")
    np.testing.assert_allclose(out.values, field.mean("time").values)
    np.testing.assert_array_equal(out.coords["lat"].values, field.coords["lat"].values)


def test_reduce_median_via_adapter(field) -> None:
    out = xnx.reduce("time lat lon -> lat lon", field, op="median")
    np.testing.assert_allclose(out.values, field.median("time").values)


def test_reduce_callable_op(field) -> None:
    out = xnx.reduce("time lat lon -> lat lon", field, op=np.sum)
    np.testing.assert_allclose(out.values, field.sum("time").values)


def test_reduce_unknown_op_raises(field) -> None:
    with pytest.raises(PatternError, match="Unknown reduce op"):
        xnx.reduce("time lat lon -> lat lon", field, op="kurtosis")


def test_repeat_adds_named_dim(mask) -> None:
    out = xnx.repeat("lat lon -> month lat lon", mask, month=4)
    assert out.dims == ("month", "lat", "lon")
    assert out.sizes["month"] == 4
    # Replicated across the new axis.
    for m in range(4):
        np.testing.assert_allclose(out.isel(month=m).values, mask.values)
    # New dim is unindexed unless coords supplied (D4).
    assert "month" not in out.coords


def test_repeat_with_explicit_coord(mask) -> None:
    out = xnx.repeat(
        "lat lon -> month lat lon", mask, month=3, coords={"month": [1, 2, 3]}
    )
    np.testing.assert_array_equal(out.coords["month"].values, [1, 2, 3])


def test_rearrange_split_and_merge() -> None:
    field = xr.DataArray(
        np.arange(2 * 4 * 4.0).reshape(2, 4, 4), dims=("time", "lat", "lon")
    )
    out = xnx.rearrange(
        "time (lat_blk lat_in) (lon_blk lon_in) "
        "-> time (lat_blk lon_blk) lat_in lon_in",
        field,
        lat_in=2,
        lon_in=2,
    )
    assert out.dims == ("time", "lat_blk_lon_blk", "lat_in", "lon_in")
    assert out.shape == (2, 4, 2, 2)


def test_rearrange_pure_transpose() -> None:
    field = xr.DataArray(np.arange(6.0).reshape(2, 3), dims=("a", "b"))
    out = xnx.rearrange("a b -> b a", field)
    assert out.dims == ("b", "a")
    np.testing.assert_allclose(out.values, field.values.T)


def test_pattern_rejects_ellipsis(field) -> None:
    with pytest.raises(PatternError, match="Ellipsis"):
        xnx.reduce("... lon -> ...", field, op="sum")


def test_pattern_requires_single_arrow(field) -> None:
    with pytest.raises(PatternError, match="exactly one"):
        xnx.reduce("time lat lon", field, op="sum")
