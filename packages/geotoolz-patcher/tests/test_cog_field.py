"""`CogField` — geotoolz-cloud's `CogSource` as a geopatcher `Field`.

The read engine (tile dedup, codecs, georeferencing, nodata) is tested in
geotoolz-cloud; this module tests what the Field layer adds: the
``select`` / ``select_many`` / ``aselect`` spellings, ``with_data``,
pickling back to a `CogField`, and the patchers and runners driving it.
Skipped unless the ``[cog]`` extra is installed; no network (``LocalStore``).
"""

from __future__ import annotations

import asyncio
import pickle
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window


pytest.importorskip("async_geotiff")

from geocloud.cog import CogSource
from georeader.geotensor import GeoTensor
from obstore.store import LocalStore

from geopatcher import AsyncSpatialPatcher, SpatialPatcher, spatial
from geopatcher.fields import CogField
from geopatcher.run import parallel_map


# 40 x 50 with 16-px tiles: partial edge tiles on both axes.
_H, _W = 40, 50
_TRANSFORM = Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_000_400.0)


def _write_cog(path: Path, *, bands: int = 3, nodata: float | None = -1.0) -> Path:
    data = (np.arange(bands * _H * _W) % 6000).reshape(bands, _H, _W).astype("float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=_H,
        width=_W,
        count=bands,
        dtype="float32",
        crs="EPSG:32630",
        transform=_TRANSFORM,
        nodata=nodata,
        tiled=True,
        blockxsize=16,
        blockysize=16,
        compress="deflate",
        interleave="band",
    ) as dst:
        dst.write(data)
    return path


def _open(path: Path) -> CogField:
    return CogField.open(
        f"file://{path}", store=LocalStore(prefix=str(path.parent)), path=path.name
    )


def _patcher(boundary: str = "shrink", size: int = 16) -> SpatialPatcher:
    return SpatialPatcher(
        geometry=spatial.geometry.Rectangular(size=(size, size), boundary=boundary),
        sampler=spatial.sampler.RegularStride(step=size),
        window=spatial.window.Boxcar(),
        aggregation=spatial.aggregation.OverlapAdd(),
    )


@pytest.fixture
def cog_path(tmp_path: Path) -> Path:
    return _write_cog(tmp_path / "scene.tif")


def test_is_a_cog_source_with_field_spellings(cog_path: Path) -> None:
    field = _open(cog_path)
    assert isinstance(field, CogSource)
    window = Window(3, 5, 20, 12)
    np.testing.assert_array_equal(
        np.asarray(field.select(window)), np.asarray(field.read_window(window))
    )
    batched = field.select_many([window, Window(0, 0, 8, 8)])
    assert [b.shape for b in batched] == [(3, 12, 20), (3, 8, 8)]
    assert asyncio.run(field.aselect(window)).shape == (3, 12, 20)
    assert len(asyncio.run(field.aselect_many([window]))) == 1


def test_with_data_carries_grid_and_nodata(cog_path: Path) -> None:
    field = _open(cog_path)
    restored = field.with_data(np.zeros((3, _H, _W), np.float32))
    assert isinstance(restored, GeoTensor)
    assert restored.transform == field.domain.transform
    assert restored.fill_value_default == -1


def test_pickles_back_to_a_cog_field(cog_path: Path) -> None:
    clone = pickle.loads(pickle.dumps(_open(cog_path)))
    assert type(clone) is CogField
    np.testing.assert_array_equal(
        np.asarray(clone.select(Window(0, 0, 8, 8))),
        np.asarray(_open(cog_path).select(Window(0, 0, 8, 8))),
    )


def test_split_merge_reproduces_rasterio(cog_path: Path) -> None:
    field = _open(cog_path)
    patcher = _patcher("shrink")
    patches = list(patcher.split(field))
    for p in patches:
        row, col = p.anchor
        assert p.data.transform == field.domain.transform * Affine.translation(col, row)
    merged = patcher.merge(patches, field.domain)
    with rasterio.open(cog_path) as src:
        np.testing.assert_array_equal(np.asarray(merged), src.read())


def test_pad_boundary_pads_with_nodata(cog_path: Path) -> None:
    edge = [p for p in _patcher("pad").split(_open(cog_path)) if p.anchor == (32, 48)]
    values = np.asarray(edge[0].data)
    assert values.shape == (3, 16, 16)
    assert (values[..., 8:, :] == -1).all() and (values[..., :, 2:] == -1).all()


def test_parallel_map_uses_the_batched_reads(cog_path: Path) -> None:
    field = _open(cog_path)
    patcher = _patcher("shrink", size=10)
    out = parallel_map(patcher, field, lambda d: np.asarray(d) * 2, n_workers=2)
    merged = patcher.merge(out, field.domain)
    with rasterio.open(cog_path) as src:
        np.testing.assert_array_equal(np.asarray(merged), src.read() * 2)


def test_async_patcher_reads_through_aselect(cog_path: Path) -> None:
    field = _open(cog_path)
    kwargs = {
        "geometry": spatial.geometry.Rectangular(size=(16, 16)),
        "sampler": spatial.sampler.RegularStride(step=16),
        "window": spatial.window.Boxcar(),
        "aggregation": spatial.aggregation.OverlapAdd(),
    }

    async def collect() -> list:
        return [p async for p in AsyncSpatialPatcher(**kwargs).asplit(field)]

    async_patches = asyncio.run(collect())
    sync_patches = list(SpatialPatcher(**kwargs).split(field))
    assert len(async_patches) == len(sync_patches) > 0
    for a, s in zip(async_patches, sync_patches, strict=True):
        np.testing.assert_array_equal(np.asarray(a.data), np.asarray(s.data))
