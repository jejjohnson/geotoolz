"""Tests for the spatial `Sampler` family."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor

from geotoolz.patch import (
    GridDomain,
    SpatialExplicit,
    SpatialJitteredStride,
    SpatialPoissonDisk,
    SpatialRandom,
    SpatialRectangular,
    SpatialRegularStride,
)


@pytest.fixture
def raster_domain() -> GeoTensor:
    return GeoTensor(
        values=np.zeros((1, 64, 64), dtype=np.float32),
        transform=rasterio.Affine.identity(),
        crs="EPSG:32630",
    )


@pytest.fixture
def rect() -> SpatialRectangular:
    return SpatialRectangular(size=(16, 16))


class TestSpatialRegularStride:
    def test_raster_anchor_count(
        self, raster_domain: GeoTensor, rect: SpatialRectangular
    ) -> None:
        s = SpatialRegularStride(step=16)
        anchors = list(s.anchors(raster_domain, rect))
        # 64x64 raster, 16x16 patches, stride 16 -> 4x4 = 16 anchors
        assert len(anchors) == 16
        # All anchors are valid (row, col) pairs
        assert all(0 <= r <= 48 and 0 <= c <= 48 for r, c in anchors)


class TestSpatialJitteredStride:
    def test_reproducible_seed(
        self, raster_domain: GeoTensor, rect: SpatialRectangular
    ) -> None:
        s1 = SpatialJitteredStride(step=16, jitter=0.5, seed=0)
        s2 = SpatialJitteredStride(step=16, jitter=0.5, seed=0)
        assert list(s1.anchors(raster_domain, rect)) == list(
            s2.anchors(raster_domain, rect)
        )


class TestSpatialRandom:
    def test_count_matches_request(
        self, raster_domain: GeoTensor, rect: SpatialRectangular
    ) -> None:
        s = SpatialRandom(n_samples=7, seed=42)
        anchors = list(s.anchors(raster_domain, rect))
        assert len(anchors) == 7

    def test_grid_anchors_are_dicts(self, rect: SpatialRectangular) -> None:
        gd = GridDomain(
            coords={"a": np.arange(64), "b": np.arange(64)},
        )
        s = SpatialRandom(n_samples=3, seed=0)
        anchors = list(s.anchors(gd, rect))
        assert len(anchors) == 3
        assert all(isinstance(a, dict) for a in anchors)


class TestSpatialPoissonDisk:
    def test_min_distance_invariant(
        self, raster_domain: GeoTensor, rect: SpatialRectangular
    ) -> None:
        s = SpatialPoissonDisk(min_dist=8.0, seed=0)
        anchors = list(s.anchors(raster_domain, rect))
        # Bridson samples are floats internally; the integer pixel cast can
        # shorten the integer-space distance by up to sqrt(2). Use a tolerance.
        for i, a in enumerate(anchors):
            for b in anchors[i + 1 :]:
                d = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
                assert d >= 8.0 - 2.0  # generous: sqrt(2) per coord


class TestSpatialExplicit:
    def test_yields_supplied(
        self, raster_domain: GeoTensor, rect: SpatialRectangular
    ) -> None:
        anchors = [(0, 0), (10, 5), (32, 16)]
        s = SpatialExplicit(anchors_=anchors)
        assert list(s.anchors(raster_domain, rect)) == anchors
