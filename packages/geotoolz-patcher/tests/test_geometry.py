"""Tests for `SpatialGeometry` subclasses.

We use georeader's `GeoTensor` as the concrete raster domain (it
satisfies `GeoDataBase`), and synthetic Grid/Point/Vector domains for
the others.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
import rasterio
import shapely
from georeader.geotensor import GeoTensor
from rasterio import features
from rasterio.windows import Window, transform as window_transform
from scipy.spatial import cKDTree

from geopatcher import (
    GridDomain,
    PointDomain,
    RasterField,
    SpatialBoxcar,
    SpatialExplicit,
    SpatialKNNGraph,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialPolygonIntersection,
    SpatialRadiusGraph,
    SpatialRectangular,
    SpatialSphericalCap,
)
from geopatcher._src.spatial.geometry import _haversine_km


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
        nb = g.neighborhood(grid, anchor=(0.0, 0.0))
        # #187: a {dim: slice} box + mask a grid `select` accepts, not an
        # `argwhere` array. 0.1 deg ~ 11.1 km, so the cap spans +-10 cells.
        assert nb.window == {"lat": slice(0, 21), "lon": slice(0, 21)}
        lat, lon = np.meshgrid(grid.coords["lat"], grid.coords["lon"], indexing="ij")
        expected = _haversine_km(0.0, 0.0, lat, lon) <= 120.0
        np.testing.assert_array_equal(nb.mask, expected)

    def test_grid_cap_index_anchor_split_merge(self) -> None:
        xr = pytest.importorskip("xarray")
        from geopatcher import (
            SpatialBoxcar,
            SpatialMax,
            SpatialPatcher,
            SpatialRegularStride,
            XarrayField,
        )

        # (time, lon, lat) on purpose: the mask must follow the dim order
        # and broadcast over the non-spatial dim.
        da = xr.DataArray(
            np.arange(2 * 15 * 11, dtype=float).reshape(2, 15, 11),
            dims=("time", "lon", "lat"),
            coords={
                "time": [0, 1],
                "lon": np.linspace(10.0, 11.4, 15),
                "lat": np.linspace(70.0, 71.0, 11),
            },
        )
        field = XarrayField(da)
        patcher = SpatialPatcher(
            geometry=SpatialSphericalCap(radius_km=15.0),
            sampler=SpatialRegularStride(step=(2, 5, 5)),
            window=SpatialBoxcar(),
            aggregation=SpatialMax(),
        )
        patches = list(patcher.split(field))
        assert patches
        for p in patches:
            assert p.weights.shape == p.data.shape
            centre_lat = da["lat"].values[p.anchor["lat"]]
            centre_lon = da["lon"].values[p.anchor["lon"]]
            lat = p.data["lat"].values[None, None, :]
            lon = p.data["lon"].values[None, :, None]
            inside = _haversine_km(centre_lat, centre_lon, lat, lon) <= 15.0
            np.testing.assert_array_equal(
                p.weights, np.broadcast_to(inside, p.data.shape)
            )
        merged = patcher.merge(patches, field.domain)
        assert merged.shape == da.shape

    def test_grid_cap_needs_lat_lon_dims(self) -> None:
        grid = GridDomain(coords={"y": np.arange(3.0), "x": np.arange(3.0)})
        with pytest.raises(ValueError, match="lat"):
            SpatialSphericalCap(radius_km=1.0).neighborhood(grid, {"y": 0, "x": 0})


class TestReachableCombos:
    """#187: geometry x domain combinations a sampler can actually feed."""

    def test_knn_k_greater_than_n(self, point_domain: PointDomain) -> None:
        # scipy pads k > N with the sentinel index N (= 5 here).
        idx = SpatialKNNGraph(k=10).neighborhood(point_domain, anchor=0)
        assert sorted(int(i) for i in idx) == [0, 1, 2, 3, 4]

    def test_knn_rejects_non_positive_k(self) -> None:
        with pytest.raises(ValueError, match="k must be"):
            SpatialKNNGraph(k=0)

    @pytest.mark.parametrize(
        "geometry",
        [SpatialKNNGraph(k=2), SpatialRadiusGraph(radius=1.5)],
        ids=["KNNGraph", "RadiusGraph"],
    )
    def test_random_on_vector_domain(self, geometry: Any) -> None:
        gpd = pytest.importorskip("geopandas")
        shapely = pytest.importorskip("shapely")
        from geopatcher import SpatialRandom, VectorDomain

        polys = gpd.GeoSeries(
            [shapely.box(i, 0, i + 1, 1) for i in range(5)], crs="EPSG:3857"
        )
        domain = VectorDomain(geometry=polys, sindex=polys.sindex, crs=polys.crs)
        anchors = list(SpatialRandom(n_samples=4, seed=0).anchors(domain, geometry))
        assert all(isinstance(a, int) for a in anchors)
        for a in anchors:
            idx = geometry.neighborhood(domain, a)
            # The anchor feature's centroid is at distance 0 from itself.
            assert a in {int(i) for i in idx}

    def test_rectangular_grid_size_must_match_dims(
        self, grid_domain: GridDomain
    ) -> None:
        geom = SpatialRectangular(size=(4,))
        with pytest.raises(ValueError, match="every GridDomain dim"):
            geom.neighborhood(grid_domain, {"lat": 0, "lon": 0})


# --- SpatialPolygonIntersection on a real RasterField (#184) -----------------
#
# A 40x50 UTM raster with non-square 10 m x 20 m pixels and a non-zero
# origin, so window maths that silently assume an identity transform fail.
# Polygons are written in fractional *pixel* coordinates (col, row) and
# mapped through the transform, which keeps the expected windows legible.

_POLY_H, _POLY_W = 40, 50
_POLY_T = rasterio.transform.from_origin(500_000.0, 4_000_000.0, 10.0, 20.0)


def _px_poly(coords: list[tuple[float, float]]) -> Any:
    return shapely.Polygon([_POLY_T * (c, r) for c, r in coords])


def _px_box(c0: float, r0: float, c1: float, r1: float) -> Any:
    return _px_poly([(c0, r0), (c1, r0), (c1, r1), (c0, r1)])


def _px_circle(c: float, r: float, radius: float) -> Any:
    t = np.linspace(0.0, 2 * np.pi, 64, endpoint=False)
    return _px_poly(
        list(zip(c + radius * np.cos(t), r + radius * np.sin(t), strict=True))
    )


# (id, polygon, expected (col_off, row_off, width, height) after outward
# rounding and clipping to the 40x50 domain).
_NON_ALIGNED_POLYGONS: list[tuple[str, Any, tuple[int, int, int, int]]] = [
    # The issue's case: every edge fractional.
    ("box", _px_box(10.3, 4.2, 15.7, 9.9), (10, 4, 6, 6)),
    # The rounded window's edge columns/rows have their centres *outside*.
    ("box-centres-out", _px_box(20.6, 12.6, 25.4, 17.4), (20, 12, 6, 6)),
    (
        "triangle",
        _px_poly([(30.2, 20.3), (38.7, 21.1), (33.4, 31.8)]),
        (30, 20, 9, 12),
    ),
    # Overlaps "box-centres-out" so the merge sees shared pixels.
    ("circle", _px_circle(23.0, 20.0, 4.3), (18, 15, 10, 10)),
    # Straddles the bottom-right corner of the domain.
    ("straddle-br", _px_box(45.5, 35.3, 53.2, 44.0), (45, 35, 5, 5)),
    # Straddles the top-left corner (negative pixel coordinates).
    ("straddle-tl", _px_box(-3.2, -2.5, 4.6, 3.3), (0, 0, 5, 4)),
]


def _poly_raster_field() -> RasterField:
    arr = np.arange(_POLY_H * _POLY_W, dtype=np.float32).reshape(_POLY_H, _POLY_W)
    return RasterField(GeoTensor(values=arr, transform=_POLY_T, crs="EPSG:32630"))


def _poly_patcher(polys: list[Any]) -> SpatialPatcher:
    return SpatialPatcher(
        geometry=SpatialPolygonIntersection(polygons=pd.Series(polys)),
        sampler=SpatialExplicit(anchors_=list(range(len(polys)))),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )


def _centre_membership(poly: Any) -> np.ndarray:
    """Full-grid reference: True where the pixel centre lies inside ``poly``.

    Computed with shapely on explicit pixel centres, independently of
    rasterio's rasteriser.
    """
    rows, cols = np.mgrid[0:_POLY_H, 0:_POLY_W]
    xs, ys = _POLY_T * (cols + 0.5, rows + 0.5)
    return shapely.contains_xy(poly, xs, ys)


class TestSpatialPolygonIntersectionRaster:
    @pytest.mark.parametrize(
        ("poly", "expected"),
        [(p, e) for _, p, e in _NON_ALIGNED_POLYGONS],
        ids=[name for name, _, _ in _NON_ALIGNED_POLYGONS],
    )
    def test_polygon_intersection_non_aligned_on_raster(
        self, poly: Any, expected: tuple[int, int, int, int]
    ) -> None:
        field = _poly_raster_field()
        (patch,) = list(_poly_patcher([poly]).split(field))

        window = patch.indices.window
        assert window == Window(*expected)
        assert all(
            isinstance(v, int)
            for v in (window.col_off, window.row_off, window.width, window.height)
        )
        rs, cs = window.toslices()
        # The chip is exactly the read of the rounded, in-domain window.
        np.testing.assert_array_equal(
            np.asarray(patch.data.values), field.reader.values[rs, cs]
        )
        assert patch.data.transform == window_transform(window, _POLY_T)
        # Mask cells are the raster's pixels: centre membership, sliced.
        assert patch.weights.shape == (window.height, window.width)
        np.testing.assert_array_equal(patch.weights, _centre_membership(poly)[rs, cs])
        assert patch.weights.any()

    def test_mask_matches_rasterio_full_grid(self) -> None:
        # Second reference: rasterise each polygon on the full domain grid
        # with rasterio, then slice by the returned window.
        field = _poly_raster_field()
        polys = pd.Series([p for _, p, _ in _NON_ALIGNED_POLYGONS])
        geom = SpatialPolygonIntersection(polygons=polys)
        for i, poly in enumerate(polys):
            mw = geom.neighborhood(field.domain, i)
            full = features.geometry_mask(
                [poly], out_shape=(_POLY_H, _POLY_W), transform=_POLY_T, invert=True
            )
            np.testing.assert_array_equal(mw.mask, full[mw.window.toslices()])

    def test_split_merge_round_trip(self) -> None:
        field = _poly_raster_field()
        polys = [p for _, p, _ in _NON_ALIGNED_POLYGONS]
        patcher = _poly_patcher(polys)
        merged = np.asarray(patcher.merge(patcher.split(field), field.domain))

        covered = np.logical_or.reduce([_centre_membership(p) for p in polys])
        source = field.reader.values
        assert merged.shape == source.shape
        # Every pixel whose centre lies in some polygon is reconstructed
        # exactly (overlaps average identical values); every other pixel
        # carries the aggregation's zero-weight fill.
        np.testing.assert_array_equal(merged[covered], source[covered])
        np.testing.assert_array_equal(merged[~covered], 0.0)

    def test_polygon_outside_domain_raises(self) -> None:
        field = _poly_raster_field()
        geom = SpatialPolygonIntersection(
            polygons=pd.Series([_px_box(60.5, 10.2, 70.3, 20.8)])
        )
        with pytest.raises(ValueError, match="does not overlap the raster"):
            geom.neighborhood(field.domain, 0)
