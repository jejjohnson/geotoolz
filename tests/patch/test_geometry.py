"""Tests for `SpatialGeometry` subclasses.

We use georeader's `GeoTensor` as the concrete raster domain (it
satisfies `GeoDataBase`), and synthetic Grid/Point/Vector domains for
the others.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor
from scipy.spatial import cKDTree

from geotoolz.patch import (
    GridDomain,
    PointDomain,
    SpatialKNNGraph,
    SpatialRadiusGraph,
    SpatialRectangular,
    SpatialSphericalCap,
)


@pytest.fixture
def raster_domain() -> GeoTensor:
    return GeoTensor(
        values=np.zeros((1, 100, 100), dtype=np.float32),
        transform=rasterio.Affine.translation(0, 100) * rasterio.Affine.scale(1, -1),
        crs="EPSG:32630",
    )


@pytest.fixture
def grid_domain() -> GridDomain:
    return GridDomain(
        coords={"lat": np.linspace(-90, 90, 181), "lon": np.linspace(-180, 180, 361)},
    )


@pytest.fixture
def point_domain() -> PointDomain:
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [5.0, 5.0]])
    return PointDomain(coords=coords, kdtree=cKDTree(coords))


class TestSpatialRectangular:
    def test_raster_neighborhood(self, raster_domain: GeoTensor) -> None:
        g = SpatialRectangular(size=(8, 8))
        win = g.neighborhood(raster_domain, anchor=(10, 20))
        assert int(win.row_off) == 10
        assert int(win.col_off) == 20
        assert int(win.width) == 8
        assert int(win.height) == 8

    def test_grid_neighborhood(self, grid_domain: GridDomain) -> None:
        g = SpatialRectangular(size=(16, 16))
        idx = g.neighborhood(grid_domain, anchor={"lat": 10, "lon": 20})
        assert idx == {"lat": slice(10, 26), "lon": slice(20, 36)}

    def test_unsupported_domain_raises(self, point_domain: PointDomain) -> None:
        g = SpatialRectangular(size=(8, 8))
        with pytest.raises(NotImplementedError):
            g.neighborhood(point_domain, anchor=0)


class TestSpatialKNNGraph:
    def test_returns_k_neighbors(self, point_domain: PointDomain) -> None:
        g = SpatialKNNGraph(k=3)
        idx = g.neighborhood(point_domain, anchor=np.array([0.5, 0.5]))
        assert len(idx) == 3
        # 0.5,0.5 should hit (0,0), (1,0), (0,1), (1,1) — closest three of those
        assert set(int(i) for i in idx) <= {0, 1, 2, 3}


class TestSpatialRadiusGraph:
    def test_radius_query(self, point_domain: PointDomain) -> None:
        g = SpatialRadiusGraph(radius=1.5)
        idx = g.neighborhood(point_domain, anchor=np.array([0.0, 0.0]))
        # Within radius 1.5 of (0,0): (0,0), (1,0), (0,1), (1,1)
        assert sorted(int(i) for i in idx) == [0, 1, 2, 3]


class TestSpatialSphericalCap:
    def test_grid_cap(self) -> None:
        # Tiny lat/lon grid around the equator
        grid = GridDomain(
            coords={
                "lat": np.linspace(-1, 1, 21),
                "lon": np.linspace(-1, 1, 21),
            }
        )
        g = SpatialSphericalCap(radius_km=120.0)
        idx = g.neighborhood(grid, anchor=(0.0, 0.0))
        # The 0.0,0.0 cell + immediate neighbors should be in the cap
        assert len(idx) >= 5
