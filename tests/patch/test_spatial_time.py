"""Tests for `SpatioTemporalPatcher` (product + coupled coupling)."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor

from geotoolz.patch import (
    RasterField,
    SpatialBoxcar,
    SpatialExplicit,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRectangular,
    SpatialRegularStride,
    SpatioTemporalPatch,
    SpatioTemporalPatcher,
    TemporalCausalBoxcar,
    TemporalFixedLookback,
    TemporalMean,
    TemporalPatcher,
    TemporalRegularStride,
)


@pytest.fixture
def time_field() -> RasterField:
    # (time=8, H=16, W=16); the spatial select returns the full time series.
    arr = np.arange(8 * 16 * 16, dtype=np.float32).reshape(8, 16, 16)
    gt = GeoTensor(
        values=arr,
        transform=rasterio.Affine.identity(),
        crs="EPSG:32630",
    )
    return RasterField(gt)


@pytest.fixture
def sp() -> SpatialPatcher:
    return SpatialPatcher(
        geometry=SpatialRectangular(size=(8, 8)),
        sampler=SpatialRegularStride(step=8),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )


@pytest.fixture
def tp() -> TemporalPatcher:
    return TemporalPatcher(
        geometry=TemporalFixedLookback(length=4),
        sampler=TemporalRegularStride(step=4),
        window=TemporalCausalBoxcar(),
        aggregation=TemporalMean(),
    )


class TestProductCoupling:
    def test_yields_spatial_x_temporal_patches(
        self, time_field: RasterField, sp: SpatialPatcher, tp: TemporalPatcher
    ) -> None:
        stp = SpatioTemporalPatcher(spatial=sp, temporal=tp, coupling="product")
        patches = list(stp.split(time_field))
        # Spatial: 4 (2x2) ; Temporal: 2 (8/4); Product: 8
        assert len(patches) == 8
        assert all(isinstance(p, SpatioTemporalPatch) for p in patches)
        # Each patch's data is a time-slice of the spatial chip; the temporal
        # axis is <= 4 (early anchors get a shorter lookback at the boundary).
        for p in patches:
            assert p.data.shape[1:] == (8, 8)
            assert 1 <= p.data.shape[0] <= 4


class TestCoupledCoupling:
    def test_requires_paired_anchors(self) -> None:
        # Couple coupling expects spatial.sampler.anchors_ to be (space, time)
        sp_explicit = SpatialPatcher(
            geometry=SpatialRectangular(size=(8, 8)),
            sampler=SpatialExplicit(anchors_=[((0, 0), 0), ((0, 8), 4)]),
            window=SpatialBoxcar(),
            aggregation=SpatialOverlapAdd(),
        )
        tp = TemporalPatcher(
            geometry=TemporalFixedLookback(length=2),
            sampler=TemporalRegularStride(step=1),
            window=TemporalCausalBoxcar(),
            aggregation=TemporalMean(),
        )
        stp = SpatioTemporalPatcher(
            spatial=sp_explicit, temporal=tp, coupling="coupled"
        )

        arr = np.arange(8 * 16 * 16, dtype=np.float32).reshape(8, 16, 16)
        field = RasterField(
            GeoTensor(
                values=arr,
                transform=rasterio.Affine.identity(),
                crs="EPSG:32630",
            )
        )
        patches = list(stp.split(field))
        assert len(patches) == 2
        anchors = [p.space for p in patches]
        assert (0, 0) in anchors
        assert (0, 8) in anchors

    def test_coupled_requires_anchors_(
        self, time_field: RasterField, sp: SpatialPatcher, tp: TemporalPatcher
    ) -> None:
        # sp's sampler is RegularStride, not Explicit — coupled must raise.
        stp = SpatioTemporalPatcher(spatial=sp, temporal=tp, coupling="coupled")
        with pytest.raises(TypeError, match="anchors_"):
            list(stp.split(time_field))


class TestUnknownCoupling:
    def test_raises(
        self, time_field: RasterField, sp: SpatialPatcher, tp: TemporalPatcher
    ) -> None:
        stp = SpatioTemporalPatcher(spatial=sp, temporal=tp, coupling="weird")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="unknown coupling"):
            list(stp.split(time_field))
