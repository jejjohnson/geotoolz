"""End-to-end test of the chip → predict → stitch pipeline through `Sequential`."""

from __future__ import annotations

import numpy as np
import rasterio
from georeader.geotensor import GeoTensor

from geotoolz import Sequential
from geotoolz.core import Lambda
from geotoolz.patch import (
    ApplyToChips,
    GridSampler,
    RasterField,
    SpatialBoxcar,
    SpatialHann,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRectangular,
    SpatialRegularStride,
    Stitch,
)


def _ones_field() -> RasterField:
    arr = np.ones((32, 32), dtype=np.float32)
    gt = GeoTensor(values=arr, transform=rasterio.Affine.identity(), crs="EPSG:32630")
    return RasterField(gt)


def test_sliding_window_inference_boxcar() -> None:
    """Tile -> double -> stitch with non-overlapping Boxcar windows."""
    field = _ones_field()
    patcher = SpatialPatcher(
        geometry=SpatialRectangular(size=(8, 8)),
        sampler=SpatialRegularStride(step=8),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )
    double = Lambda(lambda gt: np.asarray(gt) * 2.0, name="double")
    pipe = Sequential(
        [
            GridSampler(patcher),
            ApplyToChips(double),
            Stitch(SpatialOverlapAdd(), domain=field.reader),
        ]
    )
    result = pipe(field)
    assert result.shape == (32, 32)
    # Boxcar + non-overlapping stride: full coverage with weight=1.
    np.testing.assert_allclose(result, 2.0)


def test_sliding_window_inference_hann_overlap() -> None:
    """Hann window with stride < patch size to ensure overlap fills boundaries."""
    field = _ones_field()
    patcher = SpatialPatcher(
        geometry=SpatialRectangular(size=(8, 8)),
        sampler=SpatialRegularStride(step=4),
        window=SpatialHann(),
        aggregation=SpatialOverlapAdd(),
    )
    double = Lambda(lambda gt: np.asarray(gt) * 2.0, name="double")
    pipe = Sequential(
        [
            GridSampler(patcher),
            ApplyToChips(double),
            Stitch(SpatialOverlapAdd(), domain=field.reader),
        ]
    )
    result = pipe(field)
    # Strict interior should be covered by enough Hann patches to sum to 2.
    np.testing.assert_allclose(result[8:24, 8:24], 2.0, rtol=1e-6)
