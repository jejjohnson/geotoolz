"""The raster carrier at geopatcher's boundary: `geoarray.GeoArray` / `RasterSource`.

Raster fields take a `geoarray.RasterSource` (a `GeoArray` or a lazy
reader) and refuse georeader objects with a pointer to the adapter; a lazy
source is read window by window, never loaded whole; a `geoarray.Grid` is
a raster domain on its own.
"""

from __future__ import annotations

import json
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
    return GeoArray(values, _T, "EPSG:32630", fill_value=-1.0, attrs={"k": 1})


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
