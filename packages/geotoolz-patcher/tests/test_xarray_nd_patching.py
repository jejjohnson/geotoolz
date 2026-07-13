"""End-to-end integration: SpatialPatcher + XarrayField + IndexedPatchView.

The xrpatcher migration story in one place:

1. Wrap a 2-D labelled xarray DataArray as `XarrayField`.
2. Drive `SpatialPatcher` over it with grid-style anchors.
3. Random-access via `IndexedPatchView`.
4. Reconstruct with `merge_to_xarray` and assert round-trip identity.

We keep the cube 2-D — `SpatialRegularStride` over a `GridDomain` tiles
every coord dim, so adding a `time` axis would mean also tiling time
(which is the `TemporalPatcher` job, covered separately by
`test_temporal_stencils.py`). xrpatcher's quickstart is similarly 2-D
(``data.u[..., :240, :360]``).
"""

from __future__ import annotations

import numpy as np
import pytest


xr = pytest.importorskip("xarray")

from geopatcher import (
    IncompleteScanConfiguration,
    IndexedPatchView,
    SpatialBoxcar,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRectangular,
    SpatialRegularStride,
)
from geopatcher._src.fields.xarray import XarrayField


def _cube() -> xr.DataArray:
    """(lat=12, lon=24) labelled DataArray — non-overlapping 6x6 tiles."""
    return xr.DataArray(
        np.arange(12 * 24, dtype=np.float32).reshape(12, 24),
        dims=("latitude", "longitude"),
        coords={
            "latitude": np.linspace(-30, 30, 12),
            "longitude": np.linspace(0, 60, 24),
        },
    )


def _patcher(check_full_scan: bool = True) -> SpatialPatcher:
    return SpatialPatcher(
        geometry=SpatialRectangular(size=(6, 6)),
        sampler=SpatialRegularStride(step=(6, 6), check_full_scan=check_full_scan),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )


class TestSplitAndAggregate:
    def test_patch_count_matches_grid_tiling(self) -> None:
        # 12 / 6 = 2 along lat, 24 / 6 = 4 along lon → 8 spatial patches.
        field = XarrayField(_cube())
        patches = list(_patcher().split(field))
        assert len(patches) == 8

    def test_anchors_match_split_count(self) -> None:
        field = XarrayField(_cube())
        patcher = _patcher()
        assert len(patcher.anchors(field)) == 8


class TestIndexedView:
    def test_indexed_access_matches_split(self) -> None:
        field = XarrayField(_cube())
        patcher = _patcher()
        view = IndexedPatchView(patcher, field)
        from_split = list(patcher.split(field))
        for i, expected in enumerate(from_split):
            np.testing.assert_array_equal(
                np.asarray(view[i].data), np.asarray(expected.data)
            )

    def test_cached_view_returns_identical_object(self) -> None:
        field = XarrayField(_cube())
        view = IndexedPatchView(_patcher(), field, cache=True)
        a = view[2]
        b = view[2]
        assert a is b

    def test_preload_materialises_xarray_data(self) -> None:
        # The DataArray patches expose `.load()`; preload should call it.
        field = XarrayField(_cube())
        view = IndexedPatchView(_patcher(), field, cache=True, preload=True)
        patch = view[0]
        # `.load()` returns a DataArray whose backing is now a numpy array.
        assert isinstance(patch.data, xr.DataArray)
        assert isinstance(patch.data.data, np.ndarray)


class TestMergeToXarray:
    def test_round_trip_identity_with_boxcar_no_overlap(self) -> None:
        da = _cube()
        field = XarrayField(da)
        patcher = _patcher()
        patches = list(patcher.split(field))
        recon = patcher.merge_to_xarray(patches, field)
        np.testing.assert_allclose(recon.values, da.values)

    def test_recon_preserves_coords(self) -> None:
        da = _cube()
        field = XarrayField(da)
        patcher = _patcher()
        recon = patcher.merge_to_xarray(list(patcher.split(field)), field)
        for name in ("latitude", "longitude"):
            np.testing.assert_array_equal(recon[name].values, da[name].values)


class TestCheckFullScan:
    def test_partial_tile_raises_at_anchor_time(self) -> None:
        # 25 latitudes can't be tiled by 6 → (25 - 6) % 6 = 1.
        cube = xr.DataArray(
            np.zeros((25, 24), dtype=np.float32),
            dims=("latitude", "longitude"),
            coords={
                "latitude": np.linspace(-30, 30, 25),
                "longitude": np.linspace(0, 60, 24),
            },
        )
        field = XarrayField(cube)
        with pytest.raises(IncompleteScanConfiguration, match="latitude"):
            list(_patcher(check_full_scan=True).split(field))


class TestCoordsPerPatch:
    def test_coords_align_with_patch_indices(self) -> None:
        da = _cube()
        field = XarrayField(da)
        patcher = _patcher()
        patches = list(patcher.split(field))
        coords = field.coords_per_patch(patches)
        assert len(coords) == len(patches)
        # First patch covers the top-left 6x6 lat/lon window across all time.
        first = coords[0]
        np.testing.assert_array_equal(
            first["latitude"].values, da["latitude"].values[:6]
        )
        np.testing.assert_array_equal(
            first["longitude"].values, da["longitude"].values[:6]
        )
