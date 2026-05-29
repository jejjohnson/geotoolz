"""Tests for pack_dataset / unpack_dataset."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import xrtoolz.einx as xnx


@pytest.fixture
def ds() -> xr.Dataset:
    base = xr.DataArray(
        np.ones((2, 3)),
        dims=("lat", "lon"),
        coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0, 2.0]},
    )
    return xr.Dataset({"u": base, "v": base * 2.0})


def test_pack_unpack_round_trip(ds) -> None:
    packed = xnx.pack_dataset(ds)
    assert packed.dims == ("variable", "lat", "lon")
    np.testing.assert_array_equal(packed.coords["variable"].values, ["u", "v"])

    restored = xnx.unpack_dataset(packed)
    assert set(restored.data_vars) == {"u", "v"}
    xr.testing.assert_allclose(restored["u"], ds["u"])
    xr.testing.assert_allclose(restored["v"], ds["v"])


def test_pack_selects_and_orders_variables(ds) -> None:
    packed = xnx.pack_dataset(ds, variables=["v", "u"], new_dim="channel")
    assert packed.dims == ("channel", "lat", "lon")
    np.testing.assert_array_equal(packed.coords["channel"].values, ["v", "u"])


def test_pack_empty_raises() -> None:
    with pytest.raises(ValueError, match="no variables"):
        xnx.pack_dataset(xr.Dataset())


def test_unpack_requires_coord() -> None:
    da = xr.DataArray(np.ones((2, 2)), dims=("variable", "lat"))
    with pytest.raises(ValueError, match="no coordinate"):
        xnx.unpack_dataset(da)


def test_unpack_requires_dim() -> None:
    da = xr.DataArray(np.ones((2, 2)), dims=("lat", "lon"))
    with pytest.raises(ValueError, match="not on input"):
        xnx.unpack_dataset(da)


def test_pack_rejects_mismatched_dims() -> None:
    # Variables in one Dataset share dim-coords, but may differ in *which*
    # dims they carry; packing those would misalign under outer join.
    ds = xr.Dataset(
        {
            "u": (("lat", "lon"), np.ones((2, 3))),
            "v": (("lat",), np.ones(2)),
        }
    )
    with pytest.raises(ValueError, match="share dims"):
        xnx.pack_dataset(ds)
