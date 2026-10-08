"""End-to-end integration: SpatialPatcher + XarrayField + IndexedPatchView.

The xrpatcher migration story in one place:

1. Wrap a 2-D labelled xarray DataArray as `XarrayField`.
2. Drive `SpatialPatcher` over it with grid-style anchors.
3. Random-access via `IndexedPatchView`.
4. Reconstruct with `merge_to_xarray` and assert round-trip identity.

Most cases keep the cube 2-D — `spatial.sampler.RegularStride` over a `GridDomain`
tiles every coord dim, so a `time` axis must be named in ``size`` (use the
full time length to keep it whole). `TestNDCube` covers that 3-D path and
the error raised when ``size`` omits a dim. xrpatcher's quickstart is
similarly 2-D (``data.u[..., :240, :360]``).
"""

from __future__ import annotations

import numpy as np
import pytest


xr = pytest.importorskip("xarray")

from geopatcher import SpatialPatcher, spatial
from geopatcher._src.fields.xarray import XarrayField
from geopatcher.run import IndexedPatchView
from geopatcher.spatial.sampler import IncompleteScanConfiguration


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
        geometry=spatial.geometry.Rectangular(size=(6, 6)),
        sampler=spatial.sampler.RegularStride(
            step=(6, 6), check_full_scan=check_full_scan
        ),
        window=spatial.window.Boxcar(),
        aggregation=spatial.aggregation.OverlapAdd(),
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
        geom = spatial.geometry.Rectangular(size=(6, 6))
        stride = spatial.sampler.RegularStride(
            step=(1, 6, 6), check_full_scan=check_full_scan
        )
        with pytest.raises(ValueError, match="size must name every GridDomain dim"):
            list(stride.anchors(field.domain, geom))
        jittered = spatial.sampler.JitteredStride(step=(1, 6, 6), seed=0)
        with pytest.raises(ValueError, match="size must name every GridDomain dim"):
            list(jittered.anchors(field.domain, geom))
        rand = spatial.sampler.Random(n_samples=2, seed=0)
        with pytest.raises(ValueError, match="size must name every GridDomain dim"):
            list(rand.anchors(field.domain, geom))

    def test_3d_cube_split_merge_round_trip(self) -> None:
        da = _time_cube()
        field = XarrayField(da)
        patcher = SpatialPatcher(
            geometry=spatial.geometry.Rectangular(size=(3, 6, 6)),
            sampler=spatial.sampler.RegularStride(step=(3, 6, 6), check_full_scan=True),
            window=spatial.window.Boxcar(),
            aggregation=spatial.aggregation.OverlapAdd(),
        )
        patches = list(patcher.split(field))
        assert len(patches) == 8
        recon = patcher.merge_to_xarray(patches, field)
        np.testing.assert_allclose(recon.values, da.values)
        np.testing.assert_array_equal(recon["time"].values, da["time"].values)

    @pytest.mark.parametrize("boundary", ["shrink", "pad", "reflect"])
    def test_4d_cube_split_merge_round_trip(self, boundary: str) -> None:
        # (band, time, lat, lon) with a ragged lon edge (26 = 4 x 6 + 2):
        # every axis is patched and stitched back, coordinates included.
        da = xr.DataArray(
            np.arange(2 * 3 * 12 * 26, dtype=np.float32).reshape(2, 3, 12, 26),
            dims=("band", "time", "latitude", "longitude"),
            coords={
                "band": ["red", "nir"],
                "time": np.array(
                    ["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[ns]"
                ),
                "latitude": np.linspace(-30, 30, 12),
                "longitude": np.linspace(0, 65, 26),
            },
        )
        field = XarrayField(da)
        patcher = SpatialPatcher(
            geometry=spatial.geometry.Rectangular(size=(1, 3, 6, 6), boundary=boundary),
            sampler=spatial.sampler.RegularStride(step=(1, 3, 6, 6)),
            window=spatial.window.Boxcar(),
            aggregation=spatial.aggregation.OverlapAdd(),
        )
        patches = list(patcher.split(field))
        # 2 bands x 1 time block x 2 lat blocks x 5 lon blocks (last ragged).
        assert len(patches) == 2 * 1 * 2 * 5
        assert all(p.data.ndim == 4 for p in patches)
        recon = patcher.merge_to_xarray(patches, field)
        assert recon.dims == da.dims
        np.testing.assert_allclose(recon.values, da.values)
        for dim in da.dims:
            np.testing.assert_array_equal(recon[dim].values, da[dim].values)


def _dask_field() -> XarrayField:
    """(y=64, x=64) dask-backed cube in 16x16 chunks — one chunk per patch."""
    dask_array = pytest.importorskip("dask.array")
    da = xr.DataArray(
        dask_array.ones((64, 64), chunks=16, dtype=np.float32),
        dims=("y", "x"),
        coords={"y": np.arange(64), "x": np.arange(64)},
    )
    return XarrayField(da)


def _patcher16() -> SpatialPatcher:
    return SpatialPatcher(
        geometry=spatial.geometry.Rectangular(size=(16, 16)),
        sampler=spatial.sampler.RegularStride(step=(16, 16)),
        window=spatial.window.Boxcar(),
        aggregation=spatial.aggregation.OverlapAdd(),
    )


_CHIP_BYTES = 16 * 16 * 4  # float32


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"max_in_flight_bytes": _CHIP_BYTES},
        {"max_in_flight": 2, "max_in_flight_bytes": 10**6, "prefetch": 2},
    ],
    ids=["no-limit", "byte-limit", "slots+bytes+prefetch"],
)
@pytest.mark.parametrize("with_hooks", [False, True], ids=["no-hooks", "hooks"])
def test_split_does_not_compute_dask(kwargs: dict[str, int], with_hooks: bool) -> None:
    """`split` keeps a dask-backed field lazy (#195).

    The byte budget used to run ``np.asarray(patch.data)`` on every patch —
    even with no limit configured — computing each chip in the producer.
    """
    from dask.callbacks import Callback

    class CountTasks(Callback):
        def __init__(self) -> None:
            super().__init__()
            self.n = 0

        def _pretask(self, key: object, dask: object, state: object) -> None:
            self.n += 1

    class BytesHook:
        def __init__(self) -> None:
            self.bytes_: list[int] = []

        def on_patch_done(self, anchor: object, runtime_s: float, n: int) -> None:
            self.bytes_.append(n)

    field = _dask_field()
    hook = BytesHook()
    hooks = [hook] if with_hooks else None
    chips = []
    with CountTasks() as counter:
        for patch in _patcher16().split(field, hooks=hooks, **kwargs):
            with patch:
                chips.append(patch.data)
    assert len(chips) == 16
    assert counter.n == 0, f"split computed {counter.n} dask tasks"
    # The chip stays lazy and still reports its real (dtype x shape) size.
    assert chips[0].chunks is not None
    assert chips[0].nbytes == _CHIP_BYTES
    if with_hooks:
        assert hook.bytes_ == [_CHIP_BYTES] * 16


def test_byte_budget_counts_real_dataarray_size() -> None:
    """A DataArray chip counts its dtype x shape bytes against the budget."""
    with pytest.raises(ValueError, match="exceeding max_in_flight_bytes"):
        next(_patcher16().split(_dask_field(), max_in_flight_bytes=_CHIP_BYTES - 1))
    patch = next(_patcher16().split(_dask_field(), max_in_flight_bytes=_CHIP_BYTES))
    patch.close()
