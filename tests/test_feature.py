"""Tests for feature operators."""

from __future__ import annotations

import numpy as np
import rasterio
from georeader.geotensor import GeoTensor

import geotoolz as gz


def _gt(values: np.ndarray) -> GeoTensor:
    return GeoTensor(
        values=values,
        transform=rasterio.Affine(2.0, 0.0, 100.0, 0.0, -2.0, 200.0),
        crs="EPSG:32629",
        fill_value_default=0,
    )


def test_peak_local_max_returns_scored_points() -> None:
    image = np.zeros((5, 5), dtype=float)
    image[2, 3] = 10.0
    gt = _gt(image)

    points = gz.feature.PeakLocalMax(
        min_distance=1,
        threshold_abs=5.0,
        exclude_border=False,
    )(gt)

    assert len(points) == 1
    assert points.iloc[0]["score"] == 10.0
    assert points.crs == gt.crs


def test_canny_returns_boolean_geotensor() -> None:
    image = np.zeros((10, 10), dtype=float)
    image[:, 5:] = 1.0
    gt = _gt(image)

    edges = gz.feature.Canny(sigma=0.5)(gt)

    assert edges.shape == gt.shape
    assert edges.transform == gt.transform
    assert np.asarray(edges).dtype == bool
