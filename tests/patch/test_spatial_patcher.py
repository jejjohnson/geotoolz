"""Tests for `SpatialPatcher` — split/merge end-to-end."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor

from geotoolz.patch import (
    Patch,
    RasterField,
    SpatialBoxcar,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRectangular,
    SpatialRegularStride,
)


@pytest.fixture
def field() -> RasterField:
    # 2-D so the (row, col) slicer from _resolve_indices matches the domain
    # shape exactly. The 3-D channels-first case is exercised in test_ops.py.
    arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    gt = GeoTensor(
        values=arr,
        transform=rasterio.Affine.identity(),
        crs="EPSG:32630",
    )
    return RasterField(gt)


class TestSplit:
    def test_returns_iterator(self, field: RasterField) -> None:
        patcher = SpatialPatcher(
            geometry=SpatialRectangular(size=(16, 16)),
            sampler=SpatialRegularStride(step=16),
            window=SpatialBoxcar(),
            aggregation=SpatialOverlapAdd(),
        )
        result = patcher.split(field)
        assert isinstance(result, Iterator)
        patches = list(result)
        assert len(patches) == 16  # 4x4 tiles
        assert all(isinstance(p, Patch) for p in patches)

    def test_data_matches_indices(self, field: RasterField) -> None:
        patcher = SpatialPatcher(
            geometry=SpatialRectangular(size=(16, 16)),
            sampler=SpatialRegularStride(step=16),
            window=SpatialBoxcar(),
            aggregation=SpatialOverlapAdd(),
        )
        for patch in patcher.split(field):
            assert patch.data.shape[-2:] == (16, 16)


class TestSplitMergeRoundtrip:
    def test_identity_with_boxcar_no_overlap(self, field: RasterField) -> None:
        patcher = SpatialPatcher(
            geometry=SpatialRectangular(size=(16, 16)),
            sampler=SpatialRegularStride(step=16),
            window=SpatialBoxcar(),
            aggregation=SpatialOverlapAdd(),
        )
        patches = list(patcher.split(field))
        recon = patcher.aggregation.merge(patches, field.reader)
        np.testing.assert_allclose(recon, np.asarray(field.reader))


class TestGetConfig:
    def test_records_each_axis(self, field: RasterField) -> None:
        patcher = SpatialPatcher(
            geometry=SpatialRectangular(size=(8, 8)),
            sampler=SpatialRegularStride(step=8),
            window=SpatialBoxcar(),
            aggregation=SpatialOverlapAdd(),
        )
        cfg = patcher.get_config()
        assert cfg["geometry"]["class"] == "SpatialRectangular"
        assert cfg["sampler"]["class"] == "SpatialRegularStride"
        assert cfg["window"]["class"] == "SpatialBoxcar"
        assert cfg["aggregation"]["class"] == "SpatialOverlapAdd"
