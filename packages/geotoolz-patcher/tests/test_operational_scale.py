"""Operational-scale primitives: journals, sketches, and backpressure."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor
from rasterio.windows import Window

from geopatcher import (
    Patch,
    PatchJournal,
    RasterField,
    SpatialApproxCardinality,
    SpatialApproxMode,
    SpatialApproxQuantile,
    SpatialBoxcar,
    SpatialExplicit,
    SpatialHann,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRectangular,
    SpatialRegularStride,
    SpatialReservoir,
    SpatialStreamingHistogram,
    normalize_anchor,
)


def _patch(values: np.ndarray) -> Patch:
    return Patch(
        data=values,
        anchor=(0, 0),
        indices=Window(col_off=0, row_off=0, width=values.shape[-1], height=1),
    )


@pytest.fixture
def field(raster_field_factory) -> RasterField:
    return raster_field_factory(4)


def test_patch_with_data_preserves_metadata() -> None:
    patch = _patch(np.array([[1, 2, 3]]))
    updated = patch.with_data(np.array([[4, 5, 6]]))
    assert updated.anchor == patch.anchor
    assert updated.indices == patch.indices
    np.testing.assert_array_equal(updated.data, [[4, 5, 6]])


def test_patch_with_data_does_not_mutate_original() -> None:
    original_data = np.array([[1, 2, 3]])
    patch = _patch(original_data)
    updated = patch.with_data(np.array([[4, 5, 6]]))
    # `with_data` must return a fresh patch, leaving `patch.data` untouched.
    assert updated is not patch
    np.testing.assert_array_equal(patch.data, original_data)


def test_patch_close_is_idempotent() -> None:
    released: list[int] = []
    patch = _patch(np.array([[1, 2, 3]]))
    patch._release = lambda: released.append(1)
    patch.close()
    patch.close()
    assert released == [1]


def test_patch_journal_survives_crash_simulation(tmp_path: Path) -> None:
    """Re-opening the journal recovers committed rows even without close()."""
    journal_path = tmp_path / "journal.jsonl"
    journal = PatchJournal(str(journal_path))
    journal.commit((1, 1), status="ok", runtime_s=0.1)
    journal.commit((2, 2), status="ok", runtime_s=0.2)
    # Drop the in-memory reference without explicit close — fsync should
    # have made the rows durable.
    del journal

    reopened = PatchJournal(str(journal_path))
    assert reopened.has((1, 1))
    assert reopened.has((2, 2))
    assert reopened.pending([(1, 1), (2, 2), (3, 3)]) == [(3, 3)]


def test_patch_journal_persists_and_split_skips_completed(
    tmp_path: Path, field: RasterField
) -> None:
    patcher = SpatialPatcher(
        geometry=SpatialRectangular(size=(2, 2)),
        sampler=SpatialRegularStride(step=2),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )
    journal_path = tmp_path / "journal.jsonl"
    journal = PatchJournal(str(journal_path))
    journal.commit((0, 0), status="ok", runtime_s=0.1)

    reopened = PatchJournal(str(journal_path))
    anchors = [patch.anchor for patch in patcher.split(field, journal=reopened)]

    assert (0, 0) not in anchors
    assert set(anchors) == {(0, 2), (2, 0), (2, 2)}


def test_journal_numpy_anchors(tmp_path: Path, field: RasterField) -> None:
    # numpy scalars / ndarray rows are one anchor with their Python twin:
    # commit doesn't crash, and has / pending / resume all match.
    journal_path = tmp_path / "journal.jsonl"
    journal = PatchJournal(str(journal_path))
    journal.commit((np.int64(0), np.int64(0)), status="ok", runtime_s=0.1)
    journal.commit(np.array([0, 2]), status="ok", runtime_s=0.1)
    journal.commit((np.int64(2), np.int64(2)), status="error", runtime_s=0.1)
    assert journal.has((0, 0))
    assert journal.has((np.int32(0), np.int32(2)))
    assert journal.pending([(0, 0), (0, 2), (2, 0), (2, 2)]) == [(2, 0), (2, 2)]
    assert journal.completed() == [[0, 0], [0, 2]]

    reopened = PatchJournal(str(journal_path))
    assert reopened.completed() == [[0, 0], [0, 2]]
    assert reopened.has(np.array([0, 2]))

    # `SpatialExplicit(anchors_=np.argwhere(...))` yields ndarray rows.
    patcher = SpatialPatcher(
        geometry=SpatialRectangular(size=(2, 2)),
        sampler=SpatialExplicit(anchors_=np.argwhere(np.ones((2, 2), bool)) * 2),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )
    done = []
    for patch in patcher.split(field, journal=reopened):
        assert isinstance(patch.anchor, np.ndarray)
        reopened.commit(patch.anchor, status="ok", runtime_s=0.1)
        done.append(normalize_anchor(patch.anchor))
    assert done == [[2, 0], [2, 2]]

    resumed = PatchJournal(str(journal_path))
    assert list(patcher.split(field, journal=resumed)) == []
    assert sorted(resumed.completed()) == [[0, 0], [0, 2], [2, 0], [2, 2]]


def test_normalize_anchor_rejects_unjsonable_values() -> None:
    assert normalize_anchor({"t": np.datetime64("2024-01-01")}) == {
        "t": {"__datetime64__": "2024-01-01"}
    }
    with pytest.raises(TypeError, match="cannot journal / cache an anchor"):
        normalize_anchor((object(), 1))
    with pytest.raises(TypeError, match="keys must be strings"):
        normalize_anchor({1: 2})


def test_split_rejects_patch_larger_than_byte_budget(field: RasterField) -> None:
    patcher = SpatialPatcher(
        geometry=SpatialRectangular(size=(2, 2)),
        sampler=SpatialRegularStride(step=2),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )
    with pytest.raises(ValueError, match="exceeding max_in_flight_bytes"):
        next(patcher.split(field, max_in_flight_bytes=1))


def test_sketch_aggregations_finalize_streaming_summaries() -> None:
    patch = _patch(np.array([[1, 2, 2, 3, 4, 5, 100]], dtype=np.float64))

    quantile = SpatialApproxQuantile(q=[0.5], k=32).merge([patch], None)
    cardinality = SpatialApproxCardinality(p=8).merge([patch], None)
    mode = SpatialApproxMode(k=3).merge([patch], None)
    histogram = SpatialStreamingHistogram(bins=3).merge([patch], None)
    reservoir = SpatialReservoir(k=4, seed=0).merge([patch], None)

    assert quantile["0.5"] == pytest.approx(3.0)
    assert cardinality == pytest.approx(6, rel=0.2)
    assert 2 in mode
    assert histogram["counts"].sum() == 7
    assert len(reservoir) == 4


def test_hyperloglog_sketches_merge_disjoint_sets() -> None:
    left = SpatialApproxCardinality(p=8)
    right = SpatialApproxCardinality(p=8)
    left.update(_patch(np.arange(50)))
    right.update(_patch(np.arange(50, 100)))

    left.merge(right)

    assert left.finalize() == pytest.approx(100, rel=0.2)


@pytest.fixture
def stitch_field() -> RasterField:
    gt = GeoTensor(
        values=np.arange(16 * 16, dtype=np.float32).reshape(16, 16),
        transform=rasterio.Affine.identity(),
        crs="EPSG:32630",
    )
    return RasterField(gt)


def _stitch_patcher(aggregation: SpatialOverlapAdd) -> SpatialPatcher:
    return SpatialPatcher(
        geometry=SpatialRectangular(size=(8, 8)),
        sampler=SpatialRegularStride(step=8),
        window=SpatialBoxcar(),
        aggregation=aggregation,
    )


class TestCogWriter:
    """`SpatialOverlapAdd(writer="cog")` — the COG aggregation target (gh #15)."""

    def test_roundtrip_matches_in_memory_merge(
        self, stitch_field: RasterField, tmp_path: Path
    ) -> None:
        pytest.importorskip("zarr")  # streams via a scratch zarr store
        target = str(tmp_path / "out.tif")
        agg = SpatialOverlapAdd(streaming=True, target_path=target, writer="cog")
        patcher = _stitch_patcher(agg)
        patches = list(patcher.split(stitch_field))
        out_path = patcher.merge(patches, stitch_field.domain)
        assert out_path == target

        reference = SpatialOverlapAdd().merge(patches, stitch_field.domain)
        with rasterio.open(target) as src:
            assert src.count == 1
            assert src.profile["tiled"]
            assert src.crs is not None
            written = src.read(1)
        np.testing.assert_allclose(written, np.asarray(reference), rtol=1e-6)

    def test_cog_options_forwarded(
        self, stitch_field: RasterField, tmp_path: Path
    ) -> None:
        pytest.importorskip("zarr")  # streams via a scratch zarr store
        target = str(tmp_path / "out.tif")
        agg = SpatialOverlapAdd(
            streaming=True,
            target_path=target,
            writer="cog",
            cog={"compress": "LZW", "blocksize": 256},
        )
        patcher = _stitch_patcher(agg)
        patcher.merge(patcher.split(stitch_field), stitch_field.domain)
        with rasterio.open(target) as src:
            assert str(src.profile["compress"]).lower() == "lzw"
            assert src.profile["blockxsize"] == 256

    def test_multiband_write(self, tmp_path: Path) -> None:
        pytest.importorskip("zarr")  # streams via a scratch zarr store
        target = str(tmp_path / "rgb.tif")
        data = np.random.default_rng(0).random((3, 8, 8)).astype(np.float32)
        domain = GeoTensor(
            values=data, transform=rasterio.Affine.identity(), crs="EPSG:32630"
        )
        patch = Patch(data=data, anchor=(0, 0), indices=Window(0, 0, 8, 8))
        agg = SpatialOverlapAdd(streaming=True, target_path=target, writer="cog")
        assert agg.merge([patch], domain) == target
        with rasterio.open(target) as src:
            assert src.count == 3
            np.testing.assert_allclose(src.read(), data, rtol=1e-6)

    def test_rejects_bad_rank(self, tmp_path: Path) -> None:
        class _Domain:
            shape = (2, 2, 2, 2)

        agg = SpatialOverlapAdd(
            streaming=True, target_path=str(tmp_path / "x.tif"), writer="cog"
        )
        with pytest.raises(ValueError, match="2-D domain or a 3-D"):
            agg.merge([], _Domain())

    def test_cog_writer_valid(self, tmp_path: Path) -> None:
        # #193: `writer="cog"` used to write a plain tiled GTiff (no COG
        # layout, no overviews, no nodata) from an in-memory merge.
        pytest.importorskip("zarr")  # streams via a scratch zarr store
        values = np.random.default_rng(1).random((64, 64)).astype(np.float32)
        field = RasterField(
            GeoTensor(
                values=values,
                transform=rasterio.Affine(10.0, 0.0, 0.0, 0.0, -10.0, 640.0),
                crs="EPSG:32630",
            )
        )
        patcher = SpatialPatcher(
            geometry=SpatialRectangular(size=(16, 16)),
            sampler=SpatialRegularStride(step=8),
            window=SpatialHann(),
            aggregation=SpatialOverlapAdd(),
        )
        patches = [p.with_data(np.asarray(p.data.values)) for p in patcher.split(field)]
        reference = SpatialOverlapAdd().merge(patches, field.domain)
        target = str(tmp_path / "out.tif")
        agg = SpatialOverlapAdd(
            streaming=True, target_path=target, writer="cog", cog={"blocksize": 16}
        )
        assert agg.merge(patches, field.domain) == target
        with rasterio.open(target) as src:
            assert src.driver == "GTiff"
            assert src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG"
            assert src.profile["tiled"]
            assert src.block_shapes == [(16, 16)]
            assert src.overviews(1), "a COG carries internal overviews"
            assert np.isnan(src.nodata)
            assert src.crs == field.domain.crs
            assert src.transform == field.domain.transform
            written = src.read(1)
        # Periodic Hann leaves the leading ring at Σw = 0 → nodata (NaN).
        np.testing.assert_allclose(written, reference, rtol=1e-6, equal_nan=True)
        assert np.isnan(written[0]).all()
        # No scratch files are left beside the output.
        assert sorted(os.listdir(tmp_path)) == ["out.tif"]

    def test_cog_not_overwritten(self, tmp_path: Path) -> None:
        target = tmp_path / "out.tif"
        target.write_bytes(b"precious")
        agg = SpatialOverlapAdd(streaming=True, target_path=str(target), writer="cog")
        domain = GeoTensor(
            values=np.zeros((8, 8)), transform=rasterio.Affine.identity(), crs=None
        )
        with pytest.raises(FileExistsError, match="overwrite=True"):
            agg.merge([], domain)
        assert target.read_bytes() == b"precious"


class TestZarrSharding:
    """`SpatialOverlapAdd(shard_shape=...)` — zarr v3 sharding (gh #14)."""

    def test_sharded_output_matches_in_memory_merge(
        self, stitch_field: RasterField, tmp_path: Path
    ) -> None:
        zarr = pytest.importorskip("zarr")
        agg = SpatialOverlapAdd(
            streaming=True,
            target_path=str(tmp_path),
            chunks=(8, 8),
            shard_shape=(16, 16),
        )
        patcher = _stitch_patcher(agg)
        patches = list(patcher.split(stitch_field))
        result = patcher.merge(patches, stitch_field.domain)

        reference = SpatialOverlapAdd().merge(patches, stitch_field.domain)
        np.testing.assert_allclose(np.asarray(result[:]), reference, rtol=1e-6)

        # Fresh-process read: the store on disk is valid sharded zarr.
        reread = zarr.open(str(tmp_path / "rec.zarr"), mode="r")
        np.testing.assert_allclose(np.asarray(reread[:]), reference, rtol=1e-6)
        assert reread.shards == (16, 16)
        assert reread.chunks == (8, 8)

    def test_unsharded_and_sharded_agree(
        self, stitch_field: RasterField, tmp_path: Path
    ) -> None:
        pytest.importorskip("zarr")
        patches = list(_stitch_patcher(SpatialOverlapAdd()).split(stitch_field))
        sharded = SpatialOverlapAdd(
            streaming=True,
            target_path=str(tmp_path / "sharded"),
            chunks=(8, 8),
            shard_shape=(16, 16),
        ).merge(patches, stitch_field.domain)
        plain = SpatialOverlapAdd(
            streaming=True,
            target_path=str(tmp_path / "plain"),
            chunks=(8, 8),
        ).merge(patches, stitch_field.domain)
        np.testing.assert_allclose(np.asarray(sharded[:]), np.asarray(plain[:]))
