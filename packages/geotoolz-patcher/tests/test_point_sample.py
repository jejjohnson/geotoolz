"""Tests for `PointDomain.sample` — raster extraction at scattered points."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio

from geopatcher import PointDomain


@pytest.fixture
def raster() -> object:
    """A 4x5 raster whose pixel (r, c) holds the value r * 10 + c."""

    class _Raster:
        values = (np.arange(4)[:, None] * 10 + np.arange(5)[None, :]).astype(np.float64)
        transform = rasterio.Affine.identity()

    return _Raster()


class TestNearest:
    def test_matches_hand_indexed_reference(self, raster: object) -> None:
        # Identity transform: point (x, y) falls in pixel (row=int(y), col=int(x)).
        coords = np.array([[0.5, 0.5], [2.5, 3.5], [4.9, 0.1]])
        domain = PointDomain(coords=coords, kdtree=None)
        out = domain.sample(raster)
        np.testing.assert_allclose(out, [0.0, 32.0, 4.0])

    def test_outside_extent_is_nan(self, raster: object) -> None:
        coords = np.array([[-1.0, 0.5], [2.5, 2.5], [5.5, 0.5]])
        domain = PointDomain(coords=coords, kdtree=None)
        out = domain.sample(raster)
        assert np.isnan(out[0]) and np.isnan(out[2])
        assert out[1] == 22.0


class TestBilinear:
    def test_interpolates_between_pixel_centres(self, raster: object) -> None:
        # Halfway between the centres of pixels (3, 2) and (3, 3).
        coords = np.array([[3.0, 3.5]])
        domain = PointDomain(coords=coords, kdtree=None, interp="bilinear")
        out = domain.sample(raster)
        np.testing.assert_allclose(out, [32.5])

    def test_exact_pixel_centre_matches_nearest(self, raster: object) -> None:
        coords = np.array([[2.5, 3.5]])
        nearest = PointDomain(coords=coords, kdtree=None).sample(raster)
        bilinear = PointDomain(coords=coords, kdtree=None, interp="bilinear").sample(
            raster
        )
        np.testing.assert_allclose(nearest, bilinear)

    def test_differs_from_nearest_off_centre(self, raster: object) -> None:
        coords = np.array([[2.75, 1.25]])
        nearest = PointDomain(coords=coords, kdtree=None).sample(raster)
        bilinear = PointDomain(coords=coords, kdtree=None, interp="bilinear").sample(
            raster
        )
        assert not np.allclose(nearest, bilinear)

    def test_edge_points_clamp_instead_of_nan(self, raster: object) -> None:
        # Inside the extent but outside the outermost pixel-centre ring.
        coords = np.array([[0.1, 0.1]])
        domain = PointDomain(coords=coords, kdtree=None, interp="bilinear")
        out = domain.sample(raster)
        np.testing.assert_allclose(out, [0.0])


def test_multiband_leading_dims(raster: object) -> None:
    class _MultiBand:
        values = np.stack([raster.values, raster.values * 2.0])
        transform = raster.transform

    coords = np.array([[2.5, 3.5], [0.5, 0.5]])
    domain = PointDomain(coords=coords, kdtree=None)
    out = domain.sample(_MultiBand())
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out[0], [32.0, 0.0])
    np.testing.assert_allclose(out[1], [64.0, 0.0])


def test_invalid_interp_raises() -> None:
    with pytest.raises(ValueError, match="invalid interp"):
        PointDomain(coords=np.zeros((1, 2)), kdtree=None, interp="cubic")


def test_geotensor_input() -> None:
    from georeader.geotensor import GeoTensor

    gt = GeoTensor(
        values=np.arange(16, dtype=np.float32).reshape(4, 4),
        transform=rasterio.Affine.identity(),
        crs="EPSG:32630",
    )
    domain = PointDomain(coords=np.array([[1.5, 2.5]]), kdtree=None)
    np.testing.assert_allclose(domain.sample(gt), [9.0])
