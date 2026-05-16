"""Tests for segmentation operators."""

from __future__ import annotations

import numpy as np
import rasterio
from georeader.geotensor import GeoTensor

import geotoolz as gz


def _gt(values: np.ndarray) -> GeoTensor:
    return GeoTensor(
        values=values,
        transform=rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0),
        crs="EPSG:4326",
        fill_value_default=0,
    )


def test_slic_returns_int_labels_and_masks_nan_pixels() -> None:
    values = np.zeros((1, 8, 8), dtype=float)
    values[:, :, 4:] = 1.0
    values[:, 0, 0] = np.nan
    gt = _gt(values)

    labels = gz.segment.SLIC(n_segments=4, compactness=1.0)(gt)

    assert labels.shape == gt.shape[-2:]
    assert labels.transform == gt.transform
    assert np.asarray(labels).dtype == np.int32
    assert np.asarray(labels)[0, 0] == 0
    assert np.asarray(labels).max() > 0


def test_watershed_separates_marker_basins() -> None:
    image = _gt(np.array([[3.0, 2.0, 3.0], [2.0, 1.0, 2.0], [3.0, 2.0, 3.0]]))
    markers = np.array([[1, 0, 2], [0, 0, 0], [0, 0, 0]], dtype=np.int32)

    labels = gz.segment.Watershed(markers=markers)(image)

    assert set(np.unique(np.asarray(labels))) >= {1, 2}
