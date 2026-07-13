"""Tests for `XarrayField.coords_per_patch` — xrpatcher get_coords() port."""

from __future__ import annotations

import numpy as np
import pytest


xr = pytest.importorskip("xarray")

from geopatcher import Patch
from geopatcher._src.fields.xarray import XarrayField


def _grid_da() -> xr.DataArray:
    return xr.DataArray(
        np.arange(24 * 36, dtype=np.float32).reshape(24, 36),
        dims=("latitude", "longitude"),
        coords={
            "latitude": np.linspace(-30, 30, 24),
            "longitude": np.linspace(0, 60, 36),
        },
    )


def _patch_with_dict_indices(indexer: dict[str, slice]) -> Patch:
    return Patch(data=None, anchor=None, indices=indexer, weights=None)


def test_returns_one_dataset_per_patch() -> None:
    field = XarrayField(_grid_da())
    patches = [
        _patch_with_dict_indices({"latitude": slice(0, 6), "longitude": slice(0, 6)}),
        _patch_with_dict_indices({"latitude": slice(6, 12), "longitude": slice(0, 6)}),
        _patch_with_dict_indices({"latitude": slice(0, 6), "longitude": slice(6, 12)}),
    ]
    coords = field.coords_per_patch(patches)
    assert len(coords) == 3
    assert all(isinstance(c, xr.Dataset) for c in coords)


def test_coord_values_match_slice() -> None:
    da = _grid_da()
    field = XarrayField(da)
    patch = _patch_with_dict_indices(
        {"latitude": slice(0, 6), "longitude": slice(0, 6)}
    )
    coord = field.coords_per_patch([patch])[0]
    np.testing.assert_array_equal(coord["latitude"].values, da["latitude"].values[:6])
    np.testing.assert_array_equal(coord["longitude"].values, da["longitude"].values[:6])


def test_coord_dataset_has_no_data_variables() -> None:
    field = XarrayField(_grid_da())
    patch = _patch_with_dict_indices(
        {"latitude": slice(0, 6), "longitude": slice(0, 6)}
    )
    coord = field.coords_per_patch([patch])[0]
    # `coords.to_dataset()` produces a coords-only Dataset.
    assert len(coord.data_vars) == 0
    assert set(coord.coords) == {"latitude", "longitude"}


def test_window_indices_map_to_last_two_dims() -> None:
    # rasterio Window-style indices: (row_off, col_off, height, width)
    from rasterio.windows import Window

    field = XarrayField(_grid_da())
    patch = Patch(
        data=None,
        anchor=None,
        indices=Window(col_off=6, row_off=0, height=6, width=6),
        weights=None,
    )
    coord = field.coords_per_patch([patch])[0]
    # row_off=0 → latitude[0:6]; col_off=6 → longitude[6:12]
    np.testing.assert_array_equal(
        coord["latitude"].values, _grid_da()["latitude"].values[:6]
    )
    np.testing.assert_array_equal(
        coord["longitude"].values, _grid_da()["longitude"].values[6:12]
    )


def test_unsupported_indices_type_raises() -> None:
    field = XarrayField(_grid_da())
    patch = Patch(data=None, anchor=None, indices="not-a-supported-shape", weights=None)
    with pytest.raises(TypeError, match="Can't convert"):
        field.coords_per_patch([patch])
