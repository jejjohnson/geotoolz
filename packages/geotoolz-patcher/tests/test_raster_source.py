"""The raster carrier at geopatcher's boundary: `geoarray.GeoArray` / `RasterSource`.

Raster fields take a `geoarray.RasterSource` (a `GeoArray` or a lazy
reader) and refuse georeader objects with a pointer to the adapter; a lazy
source is read window by window, never loaded whole; a `geoarray.Grid` is
a raster domain on its own.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import rasterio
from geoarray import FrozenAttrs, GeoArray, Grid, PixelWindow, RasterSource
from georeader.geotensor import GeoTensor
from georeader.rasterio_reader import RasterioReader

from geopatcher import Patch, RasterField, SpatialPatcher, spatial
from geopatcher._src import cache as cache_module
from geopatcher.fields import AsyncRasterField, RasterDomain, ReprojectingRasterField


_T = rasterio.Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_600_000.0)


def _scene(n: int = 32, bands: int = 2) -> GeoArray:
    values = np.arange(bands * n * n, dtype=np.float32).reshape(bands, n, n)
    return GeoArray(
        values, _T, "EPSG:32630", fill_value=-1.0, attrs={"k": 1}, dims="band y x"
    )


def _patcher(size: int = 8) -> SpatialPatcher:
    return SpatialPatcher(
        geometry=spatial.geometry.Rectangular(size=(size, size)),
        sampler=spatial.sampler.RegularStride(step=size),
        window=spatial.window.Boxcar(),
        aggregation=spatial.aggregation.OverlapAdd(),
    )


class _LazySource:
    """A `RasterSource` that counts reads and refuses to load whole."""

    def __init__(self, scene: GeoArray) -> None:
        self._scene = scene
        self.windows: list[PixelWindow] = []

    @property
    def grid(self) -> Grid:
        return self._scene.grid

    @property
    def shape(self) -> tuple[int, ...]:
        return self._scene.shape

    @property
    def dtype(self) -> Any:
        return self._scene.dtype

    @property
    def fill_value(self) -> Any:
        return self._scene.fill_value

    @property
    def attrs(self) -> FrozenAttrs:
        return self._scene.attrs

    def read_window(self, window: PixelWindow, *, boundless: bool = True) -> GeoArray:
        self.windows.append(window)
        return self._scene.read_window(window, boundless=boundless)

    def load(self) -> GeoArray:
        raise AssertionError("a patcher must never load a lazy source whole")


class TestRefusesGeoreader:
    def test_geotensor_names_from_georeader(self) -> None:
        gt = GeoTensor(np.zeros((4, 4), np.float32), _T, "EPSG:32630")
        for build in (RasterField, AsyncRasterField):
            with pytest.raises(TypeError, match="from_georeader"):
                build(gt)
        with pytest.raises(TypeError, match="from_georeader"):
            ReprojectingRasterField(gt, dst_crs="EPSG:3857")

    def test_lazy_reader_names_source_from_georeader(self, tmp_path) -> None:
        path = tmp_path / "scene.tif"
        with rasterio.open(
            path, "w", driver="GTiff", height=4, width=4, count=1,
            dtype="float32", crs="EPSG:32630", transform=_T,
        ) as dst:  # fmt: skip
            dst.write(np.zeros((1, 4, 4), np.float32))
        with pytest.raises(TypeError, match="source_from_georeader"):
            RasterField(RasterioReader(str(path)))

    def test_not_a_raster_source(self) -> None:
        with pytest.raises(TypeError, match="RasterSource"):
            RasterField(np.zeros((4, 4)))


class TestLazySource:
    def test_split_reads_window_by_window(self) -> None:
        scene = _scene()
        source = _LazySource(scene)
        assert isinstance(source, RasterSource)
        field = RasterField(source)
        patcher = _patcher()

        patches = list(patcher.split(field))

        assert len(source.windows) == len(patches) == 16
        assert {w.shape for w in source.windows} == {(8, 8)}
        for patch in patches:
            assert isinstance(patch.data, GeoArray)
            r, c = patch.anchor
            np.testing.assert_array_equal(
                np.asarray(patch.data), np.asarray(scene)[:, r : r + 8, c : c + 8]
            )
            assert patch.data.grid == scene.grid.crop(
                PixelWindow(col_off=c, row_off=r, width=8, height=8)
            )

    def test_merge_to_field_rebuilds_on_the_source_grid(self) -> None:
        scene = _scene()
        field = RasterField(_LazySource(scene))
        patcher = _patcher()
        out = patcher.merge_to_field(patcher.split(field), field)
        assert isinstance(out, GeoArray)
        assert out.identical(scene)


class TestDomain:
    def test_a_raster_source_is_a_raster_domain(self) -> None:
        scene = _scene()
        assert isinstance(scene, RasterDomain)
        assert isinstance(RasterField(scene).domain, RasterDomain)

    def test_a_bare_grid_plans_the_same_anchors(self) -> None:
        scene = _scene()
        patcher = _patcher()
        on_grid = list(patcher.sampler.anchors(scene.grid, patcher.geometry))
        assert on_grid == patcher.anchors(RasterField(scene))
        window = patcher.geometry.neighborhood(scene.grid, on_grid[-1])
        assert (window.row_off, window.col_off, window.height) == (24, 24, 8)


def test_patch_cache_reads_a_legacy_geotensor_entry() -> None:
    # Entries written before the switch hold the same pixels and metadata
    # under kind "geotensor" and georeader's fill name; they decode to a
    # GeoArray, so existing caches stay usable.
    scene = _scene()
    patch = Patch(data=scene, anchor=(0, 0), indices=None)
    arrays = cache_module._encode_patch(patch)
    meta = json.loads(str(arrays["meta"]))
    meta["fill_value_default"] = meta.pop("fill_value")
    payload = {"kind": "geotensor", "values": arrays["values"], "meta": meta}

    decoded = cache_module._decode_carrier(payload)

    assert isinstance(decoded, GeoArray)
    assert decoded.identical(scene)


def test_reprojecting_field_reads_a_lazy_source_window_by_window() -> None:
    # georeader warps through the adapter's GeoData view of the source:
    # each chip reads only a crop around its footprint, never the whole.
    scene = _scene(n=64)
    source = _LazySource(scene)
    field = ReprojectingRasterField(source, dst_crs="EPSG:3857")
    reference = ReprojectingRasterField(scene, dst_crs="EPSG:3857")
    assert field.domain.grid == reference.domain.grid

    patches = list(_patcher(16).split(field))

    assert patches
    assert source.windows
    assert max(w.height for w in source.windows) < 64
    for patch, ref in zip(patches, _patcher(16).split(reference), strict=True):
        assert isinstance(patch.data, GeoArray)
        assert patch.data.grid == ref.data.grid
        np.testing.assert_array_equal(np.asarray(patch.data), np.asarray(ref.data))


def _pad_patcher(size: int = 8) -> SpatialPatcher:
    return SpatialPatcher(
        geometry=spatial.geometry.Rectangular(size=(size, size), boundary="pad"),
        sampler=spatial.sampler.RegularStride(step=size),
        window=spatial.window.Boxcar(),
        aggregation=spatial.aggregation.OverlapAdd(),
    )


def _geotensor(n: int = 32) -> GeoTensor:
    return GeoTensor(np.zeros((2, n, n), np.float32), _T, "EPSG:32630")


class TestMergeRefusesGeoreader:
    """Every merge path refuses a georeader carrier instead of dropping its grid."""

    def test_merge_refuses_geotensor_chips(self) -> None:
        field = RasterField(_scene())
        patcher = _patcher()
        chips = list(patcher.split(field))
        # A GeoTensor built straight into a Patch (with_data refuses one).
        chips[0] = Patch(
            data=_geotensor(8), anchor=chips[0].anchor, indices=chips[0].indices
        )
        with pytest.raises(TypeError, match="from_georeader"):
            patcher.merge(chips, field.domain)
        with pytest.raises(TypeError, match="from_georeader"):
            patcher.merge(iter(chips), field.domain)  # a stream, checked lazily
        with pytest.raises(TypeError, match="from_georeader"):
            patcher.merge_to_field(chips, field)

    def test_merge_refuses_a_geotensor_domain(self) -> None:
        field = RasterField(_scene())
        patcher = _patcher()
        with pytest.raises(TypeError, match="from_georeader"):
            patcher.merge(patcher.split(field), _geotensor())

    def test_async_merge_refuses_geotensor_chips(self) -> None:
        import asyncio

        field = RasterField(_scene())
        patcher = _patcher()
        patch = next(iter(patcher.split(field)))
        bad = Patch(data=_geotensor(8), anchor=patch.anchor, indices=patch.indices)

        async def stream() -> Any:
            yield bad

        with pytest.raises(TypeError, match="from_georeader"):
            asyncio.run(patcher.amerge(stream(), field.domain))

    def test_merge_patches_refuses_a_geotensor_domain_and_chips(self) -> None:
        pytest.importorskip("pipekit")
        from geopatcher.integrations.pipekit import MergePatches

        field = RasterField(_scene())
        patches = list(_patcher().split(field))
        agg = spatial.aggregation.OverlapAdd()
        with pytest.raises(TypeError, match="from_georeader"):
            MergePatches(aggregation=agg, domain=_geotensor())(patches)
        bad = [
            Patch(data=_geotensor(8), anchor=p.anchor, indices=p.indices)
            for p in patches
        ]
        with pytest.raises(TypeError, match="from_georeader"):
            MergePatches(aggregation=agg, domain=field.domain)(bad)

    def test_geoops_merge_patches_refuses_a_geotensor_domain(self) -> None:
        patch_ops = pytest.importorskip("geoops.patch_ops")
        patches = list(_patcher().split(RasterField(_scene())))
        op = patch_ops.MergePatches(
            aggregation=spatial.aggregation.OverlapAdd(), domain=_geotensor()
        )
        with pytest.raises(TypeError, match="from_georeader"):
            op(patches)

    def test_on_domain_grid_refuses_a_geotensor_domain(self) -> None:
        from geopatcher._src.spatial.patcher import on_domain_grid

        with pytest.raises(TypeError, match="from_georeader"):
            on_domain_grid(
                np.zeros((2, 32, 32)), spatial.aggregation.OverlapAdd(), _geotensor()
            )

    def test_patch_with_data_refuses_a_geotensor(self) -> None:
        from geopatcher import SpatioTemporalPatch, TemporalPatch

        patch = next(iter(_patcher().split(RasterField(_scene()))))
        for carrier in (
            patch,
            TemporalPatch(data=None, anchor=0, indices=slice(0, 1)),
            SpatioTemporalPatch(data=None, space=(0, 0), time=0),
        ):
            with pytest.raises(TypeError, match="from_georeader"):
                carrier.with_data(_geotensor(8))
            with pytest.raises(TypeError, match="from_georeader"):
                carrier.with_data({"primary": _geotensor(8)})

    def test_every_field_with_data_refuses_a_geotensor(self) -> None:
        xr = pytest.importorskip("xarray")
        pytest.importorskip("rioxarray")
        pytest.importorskip("geocloud")

        from geopatcher.fields import CogField, RioXarrayField, XarrayField

        scene = _scene()
        da = xr.DataArray(np.zeros((32, 32)), dims=("y", "x"))
        fields: list[Any] = [
            RasterField(scene),
            AsyncRasterField(_AsyncSource(scene)),
            ReprojectingRasterField(scene, dst_crs="EPSG:3857"),
            XarrayField(da),
            RioXarrayField(da.rio.write_crs("EPSG:32630").rio.write_transform(_T)),
        ]
        for field in fields:
            with pytest.raises(TypeError, match="from_georeader"):
                field.with_data(_geotensor())
        with pytest.raises(TypeError, match="from_georeader"):
            # No COG to open offline: a stand-in with the grid the method reads.
            CogField.with_data(SimpleNamespace(grid=scene.grid), _geotensor())  # type: ignore[arg-type]


class _AsyncSource(_LazySource):
    async def aread_window(
        self, window: PixelWindow, *, boundless: bool = True
    ) -> GeoArray:
        return self.read_window(window, boundless=boundless)

    async def aload(self) -> GeoArray:
        raise AssertionError("a patcher must never load a lazy source whole")


class TestPaddingNeedsAFill:
    """``boundary="pad"`` never writes pixels the chip would count as data."""

    def test_float_source_without_fill_pads_nan_and_declares_it(self) -> None:
        scene = GeoArray(
            np.ones((1, 12, 12), np.float32), _T, "EPSG:32630", dims="band y x"
        )
        patches = list(_pad_patcher().split(RasterField(scene)))
        edge = next(p for p in patches if p.anchor == (8, 8))
        chip = np.asarray(edge.data)
        assert edge.data.fill_value is not None and np.isnan(edge.data.fill_value)
        assert np.isnan(chip[:, 4:, :]).all() and np.isnan(chip[:, :, 4:]).all()
        assert (chip[:, :4, :4] == 1).all()
        assert not edge.data.valid_mask()[..., 4:, :].any()
        inner = next(p for p in patches if p.anchor == (0, 0))
        assert inner.data.fill_value is None  # nothing padded, nothing declared

    @pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.bool_])
    def test_integer_source_without_fill_raises(self, dtype: Any) -> None:
        scene = GeoArray(np.ones((1, 12, 12), dtype), _T, "EPSG:32630", dims="band y x")
        with pytest.raises(ValueError, match="fill_value=None"):
            list(_pad_patcher().split(RasterField(scene)))

    def test_integer_source_without_fill_raises_on_the_async_path(self) -> None:
        import asyncio

        scene = GeoArray(
            np.ones((1, 12, 12), np.uint16), _T, "EPSG:32630", dims="band y x"
        )
        field = AsyncRasterField(_AsyncSource(scene))

        async def run() -> list[Any]:
            return [p async for p in _pad_patcher().asplit(field)]

        with pytest.raises(ValueError, match="fill_value=None"):
            asyncio.run(run())

    def test_integer_source_reflects_and_pads_with_its_fill(self) -> None:
        values = np.ones((1, 12, 12), np.uint8)
        reflect = SpatialPatcher(
            geometry=spatial.geometry.Rectangular(size=(8, 8), boundary="reflect"),
            sampler=spatial.sampler.RegularStride(step=8),
            window=spatial.window.Boxcar(),
            aggregation=spatial.aggregation.OverlapAdd(),
        )
        no_fill = GeoArray(values, _T, "EPSG:32630", dims="band y x")
        assert len(list(reflect.split(RasterField(no_fill)))) == 4
        filled = GeoArray(values, _T, "EPSG:32630", fill_value=0, dims="band y x")
        edge = list(_pad_patcher().split(RasterField(filled)))[-1]
        assert edge.data.fill_value == 0
        assert (np.asarray(edge.data)[:, 4:, :] == 0).all()


class TestRewrapFill:
    """A rewrapped output declares a fill its own dtype holds."""

    def test_uint8_output_of_a_float_source_with_a_negative_fill(self) -> None:
        scene = GeoArray(
            np.zeros((2, 8, 8), np.float32),
            _T,
            "EPSG:32630",
            fill_value=-9999.0,
            dims="band y x",
        )
        out = RasterField(scene).with_data(np.ones((2, 8, 8), np.uint8))
        assert out.dtype == np.uint8
        assert out.fill_value is None

    def test_fill_kept_when_the_output_dtype_holds_it(self) -> None:
        scene = GeoArray(
            np.zeros((2, 8, 8), np.float32),
            _T,
            "EPSG:32630",
            fill_value=-9999.0,
            dims="band y x",
        )
        assert RasterField(scene).with_data(
            np.ones((2, 8, 8), np.int16)
        ).fill_value == (-9999.0)
        assert RasterField(scene).with_data(np.ones((2, 8, 8))).fill_value == -9999.0

    def test_float_output_of_an_integer_source_is_nan_filled(self) -> None:
        scene = GeoArray(
            np.zeros((1, 8, 8), np.uint8),
            _T,
            "EPSG:32630",
            fill_value=0,
            dims="band y x",
        )
        out = RasterField(scene).with_data(np.ones((1, 8, 8), np.float32))
        assert np.isnan(float(out.fill_value))

    def test_merge_patches_on_a_band_collapsing_uint8_output(self) -> None:
        pytest.importorskip("pipekit")
        from geopatcher.integrations.pipekit import MergePatches

        scene = GeoArray(
            np.zeros((2, 8, 8), np.float32),
            _T,
            "EPSG:32630",
            fill_value=-9999.0,
            dims="band y x",
        )
        patches = [
            p.with_data(np.ones((1, *np.shape(p.data)[-2:]), np.uint8))
            for p in _patcher(size=4).split(RasterField(scene))
        ]
        out = MergePatches(
            aggregation=spatial.aggregation.HardVote(n_classes=2), domain=scene
        )(patches)
        assert isinstance(out, GeoArray)
        assert out.fill_value == -1 and out.dtype.kind == "i"
