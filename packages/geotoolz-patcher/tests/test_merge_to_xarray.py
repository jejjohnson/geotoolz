"""Tests for `SpatialPatcher.merge_to_xarray` — xarray reconstruct sugar."""

from __future__ import annotations

import numpy as np
import pytest


xr = pytest.importorskip("xarray")

from geopatcher import (
    GridDomain,
    SpatialBoxcar,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRectangular,
    SpatialRegularStride,
)
from geopatcher._src.fields.xarray import XarrayField


def _da() -> xr.DataArray:
    return xr.DataArray(
        np.arange(24 * 36, dtype=np.float32).reshape(24, 36),
        dims=("latitude", "longitude"),
        coords={
            "latitude": np.linspace(-30, 30, 24),
            "longitude": np.linspace(0, 60, 36),
        },
    )


def _patcher() -> SpatialPatcher:
    return SpatialPatcher(
        geometry=SpatialRectangular(size=(6, 6)),
        sampler=SpatialRegularStride(step=(6, 6)),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )


def test_returns_dataarray_with_original_coords() -> None:
    da = _da()
    field = XarrayField(da)
    patcher = _patcher()
    patches = list(patcher.split(field))
    recon = patcher.merge_to_xarray(patches, field)
    assert isinstance(recon, xr.DataArray)
    np.testing.assert_array_equal(recon["latitude"].values, da["latitude"].values)
    np.testing.assert_array_equal(recon["longitude"].values, da["longitude"].values)


def test_round_trip_identity_with_boxcar_no_overlap() -> None:
    # SpatialBoxcar + non-overlapping stride → recon == original (modulo dtype).
    da = _da()
    field = XarrayField(da)
    patcher = _patcher()
    patches = list(patcher.split(field))
    recon = patcher.merge_to_xarray(patches, field)
    np.testing.assert_allclose(recon.values, da.values)


def test_non_xarray_field_raises_typeerror() -> None:
    patcher = _patcher()

    class _NotAField:
        @property
        def domain(self):
            return GridDomain(coords={"x": np.arange(6)})

    with pytest.raises(TypeError, match="needs a field with `with_data`"):
        patcher.merge_to_xarray([], _NotAField())


def test_with_data_must_expose_da() -> None:
    patcher = _patcher()

    class _BadField:
        @property
        def domain(self):
            return GridDomain(
                coords={
                    "latitude": np.arange(24),
                    "longitude": np.arange(36),
                }
            )

        def with_data(self, array):
            return array  # returns a bare ndarray, not a wrapper with .da

    with pytest.raises(TypeError, match="must return a wrapper"):
        patcher.merge_to_xarray([], _BadField())
