"""End-to-end integration: SpatialPatcher + XarrayField + IndexedPatchView.

The xrpatcher migration story in one place:

1. Wrap a 2-D labelled xarray DataArray as `XarrayField`.
2. Drive `SpatialPatcher` over it with grid-style anchors.
3. Random-access via `IndexedPatchView`.
4. Reconstruct with `merge_to_xarray` and assert round-trip identity.

Most cases keep the cube 2-D — `SpatialRegularStride` over a `GridDomain`
tiles every coord dim, so a `time` axis must be named in ``size`` (use the
full time length to keep it whole). `TestNDCube` covers that 3-D path and
the error raised when ``size`` omits a dim. xrpatcher's quickstart is
similarly 2-D (``data.u[..., :240, :360]``).
"""

from __future__ import annotations

import numpy as np
import pytest


xr = pytest.importorskip("xarray")

from geopatcher import (
    IncompleteScanConfiguration,
    IndexedPatchView,
    SpatialBoxcar,
    SpatialJitteredStride,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRandom,
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


def _time_cube() -> xr.DataArray:
    """(time=3, latitude=12, longitude=24) cube with a datetime axis."""
    return xr.DataArray(
        np.arange(3 * 12 * 24, dtype=np.float32).reshape(3, 12, 24),
        dims=("time", "latitude", "longitude"),
        coords={
            "time": np.array(
                ["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[ns]"
            ),
            "latitude": np.linspace(-30, 30, 12),
            "longitude": np.linspace(0, 60, 24),
        },
    )


class TestNDCube:
    @pytest.mark.parametrize("check_full_scan", [False, True])
    def test_3d_cube_size_validation(self, check_full_scan: bool) -> None:
        # A 2-D size on a 3-D cube is ambiguous — the sampler must say so
        # instead of failing with a bare zip() length mismatch.
        field = XarrayField(_time_cube())
        geom = SpatialRectangular(size=(6, 6))
        stride = SpatialRegularStride(step=(1, 6, 6), check_full_scan=check_full_scan)
        with pytest.raises(ValueError, match="size must name every GridDomain dim"):
            list(stride.anchors(field.domain, geom))
        jittered = SpatialJitteredStride(step=(1, 6, 6), seed=0)
        with pytest.raises(ValueError, match="size must name every GridDomain dim"):
            list(jittered.anchors(field.domain, geom))
        rand = SpatialRandom(n_samples=2, seed=0)
        with pytest.raises(ValueError, match="size must name every GridDomain dim"):
            list(rand.anchors(field.domain, geom))

    def test_3d_cube_split_merge_round_trip(self) -> None:
        da = _time_cube()
        field = XarrayField(da)
        patcher = SpatialPatcher(
            geometry=SpatialRectangular(size=(3, 6, 6)),
            sampler=SpatialRegularStride(step=(3, 6, 6), check_full_scan=True),
            window=SpatialBoxcar(),
            aggregation=SpatialOverlapAdd(),
        )
        patches = list(patcher.split(field))
        assert len(patches) == 8
        recon = patcher.merge_to_xarray(patches, field)
        np.testing.assert_allclose(recon.values, da.values)
        np.testing.assert_array_equal(recon["time"].values, da["time"].values)
