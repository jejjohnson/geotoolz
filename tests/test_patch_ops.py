"""Tests for the operator wrappers (`GridSampler`, `ApplyToChips`, `Stitch`).

The wrappers re-export `geopatcher` primitives at runtime; skip cleanly
when the optional ``[patch]`` extra (which pulls in geopatcher) isn't
installed.
"""

from __future__ import annotations

import pytest


pytest.importorskip(
    "geopatcher",
    reason="geotoolz.patch_ops bridge requires the [patch] extra (geopatcher)",
)

import numpy as np
import rasterio
from geopatcher import (
    Patch,
    RasterField,
    SpatialBoxcar,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRectangular,
    SpatialRegularStride,
)
from georeader.geotensor import GeoTensor

from geotoolz import Sequential
from geotoolz.core import Lambda
from geotoolz.patch_ops import (
    ApplyToChips,
    GridSampler,
    Stitch,
)


@pytest.fixture
def field() -> RasterField:
    # 2-D field so OverlapAdd's row/col slicer matches the domain shape.
    arr = np.ones((16, 16), dtype=np.float32)
    gt = GeoTensor(
        values=arr,
        transform=rasterio.Affine.identity(),
        crs="EPSG:32630",
    )
    return RasterField(gt)


@pytest.fixture
def patcher() -> SpatialPatcher:
    return SpatialPatcher(
        geometry=SpatialRectangular(size=(8, 8)),
        sampler=SpatialRegularStride(step=8),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )


class TestGridSampler:
    def test_returns_list_of_patches(
        self, field: RasterField, patcher: SpatialPatcher
    ) -> None:
        op = GridSampler(patcher)
        patches = op(field)
        assert isinstance(patches, list)
        assert all(isinstance(p, Patch) for p in patches)
        assert len(patches) == 4  # 2x2 tiles


class TestApplyToChips:
    def test_each_chip_runs_through_operator(
        self, field: RasterField, patcher: SpatialPatcher
    ) -> None:
        patches = list(patcher.split(field))
        double = Lambda(lambda gt: np.asarray(gt) * 2.0, name="double")
        out = ApplyToChips(double)(patches)
        assert len(out) == len(patches)
        for src, dst in zip(patches, out, strict=True):
            assert dst.anchor == src.anchor
            np.testing.assert_allclose(dst.data, 2.0)


class TestStitchInSequential:
    def test_chip_predict_stitch_roundtrip(
        self, field: RasterField, patcher: SpatialPatcher
    ) -> None:
        double = Lambda(lambda gt: np.asarray(gt) * 2.0, name="double")
        pipe = Sequential(
            [
                GridSampler(patcher),
                ApplyToChips(double),
                Stitch(SpatialOverlapAdd(), domain=field.reader),
            ]
        )
        result = pipe(field)
        np.testing.assert_allclose(result, 2.0)
