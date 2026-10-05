"""Tests for `PatchCache` — content-addressed on-disk patch cache (issue #24).

The core contract: a second `split` (or a second *process*) with the
same field + geometry + window config performs zero source reads and
reconstructs each patch bit-identically.
"""

from __future__ import annotations

import datetime
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import rasterio
import xarray as xr
from _helpers import make_raster_field
from georeader.geotensor import GeoTensor
from rasterio.transform import from_origin

from geopatcher import (
    PatchCache,
    RasterField,
    SpatialBoxcar,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRandom,
    SpatialRectangular,
    SpatialRegularStride,
)


class _CountingField:
    """Wrap a `Field`, counting every `select` (source read)."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.selects = 0

    @property
    def domain(self) -> Any:
        return self.inner.domain

    def select(self, window: Any) -> Any:
        self.selects += 1
        return self.inner.select(window)

    def with_data(self, array: Any) -> Any:
        return self.inner.with_data(array)

    def __getattr__(self, name: str) -> Any:
        # Transparent wrapper: identity attributes (`reader`, `da`, `url`,
        # `cache_id`) resolve on the wrapped field.
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)


def _patcher(size: int = 8, step: int = 8) -> SpatialPatcher:
    return SpatialPatcher(
        geometry=SpatialRectangular(size=(size, size)),
        sampler=SpatialRegularStride(step=step),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )


class TestHitMiss:
    def test_second_split_reads_nothing(self, tmp_path) -> None:
        base = make_raster_field(16)  # 4 patches at size/step 8
        cache = PatchCache(tmp_path, field_id="scene")
        patcher = _patcher()

        first = _CountingField(base)
        p1 = list(patcher.split(first, cache=cache))
        assert first.selects == 4  # cold cache → every patch is read

        second = _CountingField(base)
        p2 = list(patcher.split(second, cache=cache))
        assert second.selects == 0  # warm cache → zero source reads
        assert len(p2) == len(p1) == 4
        stats = cache.stats()
        assert stats["hits"] == 4
        assert stats["entries"] == 4

    def test_config_change_is_a_miss(self, tmp_path) -> None:
        base = make_raster_field(16)
        cache = PatchCache(tmp_path, field_id="scene")
        list(_patcher(size=8, step=8).split(base, cache=cache))

        # A different geometry config → different key → cold reads again.
        other = _CountingField(base)
        list(_patcher(size=4, step=4).split(other, cache=cache))
        assert other.selects > 0

    def test_roundtrip_is_bit_identical(self, tmp_path) -> None:
        base = make_raster_field(16)
        cache = PatchCache(tmp_path, field_id="scene")
        patcher = _patcher()

        reference = {p.anchor: p for p in patcher.split(base)}  # uncached
        list(patcher.split(base, cache=cache))  # fill
        cached = {p.anchor: p for p in patcher.split(base, cache=cache)}  # hits

        for anchor, ref in reference.items():
            got = cached[anchor]
            np.testing.assert_array_equal(
                np.asarray(got.data.values), np.asarray(ref.data.values)
            )
            assert got.data.transform == ref.data.transform
            assert str(got.data.crs) == str(ref.data.crs)
            np.testing.assert_array_equal(
                np.asarray(got.weights), np.asarray(ref.weights)
            )
            assert got.indices == ref.indices


class TestFieldIdentity:
    def test_in_memory_field_without_id_raises(self, tmp_path) -> None:
        base = make_raster_field(16)  # RasterField(GeoTensor): no path/url
        cache = PatchCache(tmp_path)
        with pytest.raises(ValueError, match="field_id"):
            list(_patcher().split(base, cache=cache))

    def test_in_memory_field_with_id_caches(self, tmp_path) -> None:
        base = make_raster_field(16)
        cache = PatchCache(tmp_path, field_id="mem")
        first = _CountingField(base)
        list(_patcher().split(first, cache=cache))
        assert first.selects == 4
        second = _CountingField(base)
        list(_patcher().split(second, cache=cache))
        assert second.selects == 0

    def test_field_id_tracks_source_changes(self, tmp_path) -> None:
        # A path-backed reader's identity folds in realpath + mtime + size,
        # so editing the source invalidates its entries.
        src = tmp_path / "scene.dat"
        src.write_bytes(b"aaaa")
        os.utime(src, (1000, 1000))
        fld = SimpleNamespace(reader=SimpleNamespace(paths=[str(src)]))
        cache = PatchCache(tmp_path / "cache")
        before = cache.field_id_for(fld)

        src.write_bytes(b"bbbbbbbb")  # size + content change
        os.utime(src, (2000, 2000))  # mtime change
        after = cache.field_id_for(fld)
        assert before != after
        assert before.startswith("path:")


class TestStochasticSamplers:
    def test_seeded_random_hits_on_rerun(self, tmp_path) -> None:
        base = make_raster_field(32)
        cache = PatchCache(tmp_path, field_id="scene")
        patcher = SpatialPatcher(
            geometry=SpatialRectangular(size=(8, 8)),
            sampler=SpatialRandom(n_samples=12, seed=42),
            window=SpatialBoxcar(),
            aggregation=SpatialOverlapAdd(),
        )
        first = _CountingField(base)
        list(patcher.split(first, cache=cache))
        assert first.selects == 12
        # Same seed → identical anchors → 100% hits, zero reads.
        second = _CountingField(base)
        list(patcher.split(second, cache=cache))
        assert second.selects == 0


class TestEviction:
    def test_lru_eviction_bounds_total_size(self, tmp_path) -> None:
        base = make_raster_field(64)  # 64 patches at size/step 8
        # A cap that only fits a handful of entries.
        cache = PatchCache(tmp_path, max_bytes=4_000, field_id="scene")
        list(_patcher().split(base, cache=cache))
        stats = cache.stats()
        assert stats["bytes"] <= 4_000
        assert 0 < stats["entries"] < 64

    def test_clear_empties_the_cache(self, tmp_path) -> None:
        base = make_raster_field(16)
        cache = PatchCache(tmp_path, field_id="scene")
        list(_patcher().split(base, cache=cache))
        assert cache.stats()["entries"] == 4
        cache.clear()
        assert cache.stats()["entries"] == 0
        assert cache.stats()["hits"] == 0


class TestConcurrency:
    def test_concurrent_access_no_corruption(self, tmp_path) -> None:
        # Multiple readers/writers on one cache dir: atomic renames mean no
        # torn entries. Assert every patch round-trips correctly.
        base = make_raster_field(32)
        cache = PatchCache(tmp_path, field_id="scene")
        patcher = _patcher()
        reference = {p.anchor: np.asarray(p.data.values) for p in patcher.split(base)}

        errors: list[Exception] = []

        def worker() -> None:
            try:
                for p in patcher.split(base, cache=cache):
                    np.testing.assert_array_equal(
                        np.asarray(p.data.values), reference[p.anchor]
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert cache.stats()["entries"] == len(reference)


class TestIndexedViewIntegration:
    def test_indexed_view_uses_disk_cache(self, tmp_path) -> None:
        from geopatcher import IndexedPatchView

        base = make_raster_field(16)
        cache = PatchCache(tmp_path, field_id="scene")
        patcher = _patcher()

        first = _CountingField(base)
        view = IndexedPatchView(patcher, first, cache=cache)
        _ = [view[i] for i in range(len(view))]
        assert first.selects == len(view)

        second = _CountingField(base)
        view2 = IndexedPatchView(patcher, second, cache=cache)
        _ = [view2[i] for i in range(len(view2))]
        assert second.selects == 0


# ---------------------------------------------------------------------------
# gh #198 — corruption, key coverage, mask poisoning, carrier fidelity
# ---------------------------------------------------------------------------


def _entries(root: Any) -> list[Path]:
    return sorted(Path(root).rglob("*.npz"))


def _write_tif(
    path: Path,
    data: np.ndarray,
    *,
    crs: str = "EPSG:32630",
    nodata: float | None = None,
    **profile: Any,
) -> Path:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[-2],
        width=data.shape[-1],
        count=data.shape[0],
        dtype=data.dtype,
        crs=crs,
        transform=from_origin(500_000, 4_000_000 + 10 * data.shape[-2], 10, 10),
        nodata=nodata,
        **profile,
    ) as dst:
        dst.write(data)
    return path


def _assert_patches_equal(got: list[Any], ref: list[Any]) -> None:
    assert [p.anchor for p in got] == [p.anchor for p in ref]
    for g, r in zip(got, ref, strict=True):
        np.testing.assert_array_equal(np.asarray(g.data), np.asarray(r.data))
        assert np.asarray(g.data).dtype == np.asarray(r.data).dtype


@pytest.mark.parametrize("damage", ["bad_zip", "zero_byte", "truncated"])
def test_corrupt_entry_is_repaired(tmp_path, damage: str) -> None:
    # A torn / empty entry is a miss (never an exception out of split) and
    # is rewritten, so the run after the repair is all hits again.
    base = make_raster_field(16)
    cache = PatchCache(tmp_path, field_id="scene")
    patcher = _patcher()
    reference = list(patcher.split(base))
    list(patcher.split(base, cache=cache))
    for path in _entries(tmp_path):
        if damage == "bad_zip":
            path.write_bytes(b"this is not a zip archive")
        elif damage == "zero_byte":
            path.write_bytes(b"")
        else:
            blob = path.read_bytes()
            path.write_bytes(blob[: len(blob) // 2])

    repair = _CountingField(base)
    _assert_patches_equal(list(patcher.split(repair, cache=cache)), reference)
    assert repair.selects == 4  # every damaged entry was a miss

    warm = _CountingField(base)
    _assert_patches_equal(list(patcher.split(warm, cache=cache)), reference)
    assert warm.selects == 0  # the damaged entries were rewritten
    assert all(path.stat().st_size > 0 for path in _entries(tmp_path))
    assert not list(tmp_path.rglob("*.tmp"))  # no scratch files left behind


def test_band_subset_and_reprojection_keys_differ(tmp_path) -> None:
    from georeader.rasterio_reader import RasterioReader

    from geopatcher import ReprojectingRasterField

    tif = _write_tif(
        tmp_path / "scene.tif",
        np.arange(3 * 16 * 16, dtype=np.int16).reshape(3, 16, 16),
    )
    cache = PatchCache(tmp_path / "cache")
    patcher = _patcher()

    # Band subset: `indexes=[2]` must not be served the 3-band chips.
    list(patcher.split(RasterField(RasterioReader(str(tif))), cache=cache))
    hits = cache.stats()["hits"]
    band2 = RasterField(RasterioReader(str(tif), indexes=[2]))
    got = list(patcher.split(band2, cache=cache))
    assert cache.stats()["hits"] == hits  # all misses
    assert all(p.data.shape == (1, 8, 8) for p in got)
    _assert_patches_equal(got, list(patcher.split(band2)))

    # Reprojection: dst_crs and resampling are both part of the key.
    reader = RasterioReader(str(tif))
    utm = ReprojectingRasterField(reader, dst_crs="EPSG:32630", resampling="nearest")
    wgs = ReprojectingRasterField(reader, dst_crs="EPSG:4326", resampling="nearest")
    bil = ReprojectingRasterField(reader, dst_crs="EPSG:32630", resampling="bilinear")
    assert len({f.cache_id() for f in (utm, wgs, bil)}) == 3
    assert len({cache.field_id_for(f) for f in (utm, wgs, bil)}) == 3
    list(patcher.split(utm, cache=cache))
    for other in (wgs, bil):
        hits = cache.stats()["hits"]
        got = list(patcher.split(other, cache=cache))
        assert cache.stats()["hits"] == hits  # never served utm's chips
        _assert_patches_equal(got, list(patcher.split(other)))


def test_obstore_cog_identity_covers_store_path_and_ifd(tmp_path) -> None:
    pytest.importorskip("obstore")
    pytest.importorskip("async_tiff")
    from obstore.store import LocalStore
    from rasterio.enums import Resampling

    from geopatcher.fields import ObstoreCogField

    for folder, offset in (("a", 0), ("b", 1000)):
        (tmp_path / folder).mkdir()
        path = _write_tif(
            tmp_path / folder / "cog.tif",
            np.arange(64 * 64, dtype=np.uint16).reshape(1, 64, 64) + offset,
            tiled=True,
            blockxsize=16,
            blockysize=16,
            compress="deflate",
        )
        with rasterio.open(path, "r+") as dst:
            dst.build_overviews([2], Resampling.nearest)

    def open_cog(folder: str, ifd_index: int = 0) -> Any:
        # Same (arbitrary) url label: with an explicit store the url
        # names nothing, so it must not be the identity.
        return ObstoreCogField.from_url(
            "file:///label/cog.tif",
            store=LocalStore(prefix=str(tmp_path / folder)),
            path="cog.tif",
            ifd_index=ifd_index,
        )

    cache = PatchCache(tmp_path / "cache")
    a0, b0, a1 = open_cog("a"), open_cog("b"), open_cog("a", ifd_index=1)
    assert len({a0.cache_id(), b0.cache_id(), a1.cache_id()}) == 3
    assert len({cache.field_id_for(f) for f in (a0, b0, a1)}) == 3

    patcher = _patcher(size=16, step=16)
    list(patcher.split(a0, cache=cache))
    hits = cache.stats()["hits"]
    got = list(patcher.split(b0, cache=cache))
    assert cache.stats()["hits"] == hits
    _assert_patches_equal(got, list(patcher.split(open_cog("b"))))


class _FlakyField(_CountingField):
    """`select` raises while ``broken`` is set (a transient source outage)."""

    def __init__(self, inner: Any, *, broken: bool) -> None:
        super().__init__(inner)
        self.broken = broken

    def select(self, window: Any) -> Any:
        self.selects += 1
        if self.broken:
            raise OSError("source unavailable")
        return self.inner.select(window)


def test_mask_patches_not_cached(tmp_path) -> None:
    base = make_raster_field(16)
    cache = PatchCache(tmp_path, field_id="scene")
    patcher = SpatialPatcher(
        geometry=SpatialRectangular(size=(8, 8)),
        sampler=SpatialRegularStride(step=8),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
        on_error="mask",
    )
    masked = list(patcher.split(_FlakyField(base, broken=True), cache=cache))
    assert len(masked) == 4
    assert all(np.isnan(np.asarray(p.data)).all() for p in masked)
    assert cache.stats()["entries"] == 0  # placeholders are never stored

    recovered = _FlakyField(base, broken=False)
    got = list(patcher.split(recovered, cache=cache))
    assert recovered.selects == 4  # the source is read again after recovery
    _assert_patches_equal(got, list(_patcher().split(base)))
    assert cache.stats()["entries"] == 4


def _assert_same_attrs(got: dict, ref: dict) -> None:
    assert list(got) == list(ref)
    for key, value in ref.items():
        assert type(got[key]) is type(value), key
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(got[key], value)
            assert got[key].dtype == value.dtype
        else:
            assert got[key] == value, key


def _assert_same_carrier(got: Any, ref: Any) -> None:
    """Patch data equal in type, values, dtype and every piece of metadata."""
    assert type(got) is type(ref)
    if isinstance(ref, xr.DataArray):
        xr.testing.assert_identical(got, ref)
        assert got.dtype == ref.dtype
        _assert_same_attrs(got.attrs, ref.attrs)
        assert got.encoding == ref.encoding
        for name in ref.coords:
            _assert_same_attrs(got[name].attrs, ref[name].attrs)
            assert got[name].dtype == ref[name].dtype
        if "spatial_ref" in ref.coords:
            assert got.rio.transform() == ref.rio.transform()
            assert got.rio.crs == ref.rio.crs
            assert got.rio.nodata == ref.rio.nodata
            assert type(got.rio.nodata) is type(ref.rio.nodata)
        return
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))
    assert got.dtype == ref.dtype
    assert got.transform == ref.transform
    assert got.crs == ref.crs
    assert str(got.crs) == str(ref.crs)
    assert got.fill_value_default == ref.fill_value_default
    assert type(got.fill_value_default) is type(ref.fill_value_default)
    _assert_same_attrs(got.attrs, ref.attrs)


def _geotensor_field(tmp_path: Path) -> tuple[Any, PatchCache, SpatialPatcher]:
    gt = GeoTensor(
        values=np.arange(2 * 16 * 16, dtype=np.int16).reshape(2, 16, 16),
        transform=rasterio.Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_000_160.0),
        crs=rasterio.crs.CRS.from_epsg(32630),
        fill_value_default=-9999,
        attrs={
            "sensor": "toy",
            "wavelengths": np.array([665.0, 842.0]),
            "scale": (1, 2.5),
            "gain": np.float32(0.5),
        },
    )
    return RasterField(gt), PatchCache(tmp_path, field_id="gt"), _patcher()


def _rio_field(tmp_path: Path) -> tuple[Any, PatchCache, SpatialPatcher]:
    rioxarray = pytest.importorskip("rioxarray")
    from geopatcher.fields import RioXarrayField

    tif = _write_tif(
        tmp_path / "rio.tif",
        np.arange(2 * 16 * 16, dtype=np.int16).reshape(2, 16, 16),
        nodata=-9999,
    )
    # File-backed: identity comes from `encoding["source"]`, no field_id.
    field = RioXarrayField(rioxarray.open_rasterio(tif))
    return field, PatchCache(tmp_path / "c"), _patcher()


def _xarray_field(tmp_path: Path) -> tuple[Any, PatchCache, SpatialPatcher]:
    from geopatcher.fields import XarrayField

    da = xr.DataArray(
        np.arange(3 * 16 * 16, dtype=np.float32).reshape(3, 16, 16),
        dims=("time", "y", "x"),
        coords={
            "time": np.array(
                ["2024-01-01", "2024-01-02", "2024-01-03"], dtype="datetime64[ns]"
            ),
            "y": ("y", np.linspace(10.0, 0.0, 16), {"units": "m"}),
            "x": np.arange(16) * 0.5,
            "label": ("time", np.array(["a", "b", "c"])),
        },
        name="tas",
        attrs={"units": "K", "valid_range": np.array([0.0, 400.0], np.float32)},
    )
    patcher = SpatialPatcher(
        geometry=SpatialRectangular(size=(2, 8, 8)),
        sampler=SpatialRegularStride(step=(1, 8, 8)),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )
    return XarrayField(da), PatchCache(tmp_path, field_id="cube"), patcher


@pytest.mark.parametrize(
    "make_field",
    [_geotensor_field, _rio_field, _xarray_field],
    ids=["geotensor", "rioxarray", "xarray"],
)
def test_hit_preserves_fill_and_carrier(tmp_path, make_field) -> None:
    field, cache, patcher = make_field(tmp_path)
    uncached = list(patcher.split(field))
    list(patcher.split(field, cache=cache))  # fill
    counting = _CountingField(field)
    hits = list(patcher.split(counting, cache=cache))
    assert counting.selects == 0  # every patch below is a cache hit
    assert cache.stats()["hits"] == len(hits) == len(uncached) > 0
    for got, ref in zip(hits, uncached, strict=True):
        assert got.anchor == ref.anchor
        _assert_same_carrier(got.data, ref.data)
        np.testing.assert_array_equal(np.asarray(got.weights), np.asarray(ref.weights))


def test_unroundtrippable_carrier_is_refused(tmp_path) -> None:
    field, cache, patcher = _geotensor_field(tmp_path)
    field.reader.attrs["acquired"] = datetime.datetime(2024, 1, 1)
    with pytest.raises(TypeError, match=r"cannot store GeoTensor\.attrs"):
        list(patcher.split(field, cache=cache))
    assert cache.stats()["entries"] == 0


def test_variables_of_one_file_have_distinct_keys(tmp_path) -> None:
    """Two variables of one netCDF share source, dims, coords, shape, dtype."""
    pytest.importorskip("netCDF4")
    from geopatcher.fields import XarrayField

    coords = {"y": np.arange(16.0), "x": np.arange(16.0)}
    base = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)
    path = tmp_path / "two_vars.nc"
    xr.Dataset(
        {
            "temperature": (("y", "x"), base),
            "precipitation": (("y", "x"), base + 1000),
        },
        coords=coords,
    ).to_netcdf(path)

    patcher = SpatialPatcher(
        geometry=SpatialRectangular(size=(8, 8)),
        sampler=SpatialRegularStride(step=8),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )
    cache = PatchCache(tmp_path / "cache")
    with xr.open_dataset(path) as ds:
        temp, prcp = XarrayField(ds["temperature"]), XarrayField(ds["precipitation"])
        assert cache.field_id_for(temp) != cache.field_id_for(prcp)
        list(patcher.split(temp, cache=cache))
        counting = _CountingField(prcp)
        got = list(patcher.split(counting, cache=cache))
        assert counting.selects == len(got)  # no hit served temperature
        np.testing.assert_array_equal(np.asarray(got[0].data), base[:8, :8] + 1000)


def test_structured_dtype_metadata_is_refused(tmp_path) -> None:
    field, cache, patcher = _geotensor_field(tmp_path)
    record = np.zeros(1, dtype=[("a", "<i4"), ("b", "<f4")])[0]
    field.reader.attrs["record"] = record
    with pytest.raises(TypeError, match="structured or void dtype"):
        list(patcher.split(field, cache=cache))
    assert cache.stats()["entries"] == 0


def test_memory_store_needs_explicit_field_id(tmp_path) -> None:
    """A MemoryStore's contents live only in that instance: no auto identity."""
    pytest.importorskip("obstore")
    pytest.importorskip("async_tiff")
    from obstore.store import MemoryStore

    from geopatcher.fields import ObstoreCogField

    def open_cog(offset: int) -> Any:
        path = _write_tif(
            tmp_path / f"cog{offset}.tif",
            np.arange(32 * 32, dtype=np.uint16).reshape(1, 32, 32) + offset,
            tiled=True,
            blockxsize=16,
            blockysize=16,
        )
        store = MemoryStore()
        store.put("cog.tif", Path(path).read_bytes())
        return ObstoreCogField.from_url(
            "memory:///cog.tif", store=store, path="cog.tif"
        )

    a, b = open_cog(0), open_cog(1000)
    with pytest.raises(ValueError, match="Pass field_id"):
        PatchCache(tmp_path / "auto").field_id_for(a)

    patcher = _patcher(size=16, step=16)
    cache_a = PatchCache(tmp_path / "shared", field_id="scene-a")
    cache_b = PatchCache(tmp_path / "shared", field_id="scene-b")
    assert cache_a.field_id_for(a) != cache_b.field_id_for(b)
    list(patcher.split(a, cache=cache_a))
    got = list(patcher.split(b, cache=cache_b))
    assert cache_b.stats()["hits"] == 0
    _assert_patches_equal(got, list(patcher.split(open_cog(1000))))
