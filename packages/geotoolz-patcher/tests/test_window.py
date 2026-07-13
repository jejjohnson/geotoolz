"""Tests for the spatial `*Window` family."""

from __future__ import annotations

import numpy as np
import pytest

from geopatcher import (
    SpatialBoxcar,
    SpatialCustom,
    SpatialGaussian,
    SpatialHann,
    SpatialRectangular,
    SpatialTukey,
)


@pytest.fixture
def geom() -> SpatialRectangular:
    return SpatialRectangular(size=(8, 8))


class TestSpatialBoxcar:
    def test_uniform(self, geom: SpatialRectangular) -> None:
        w = SpatialBoxcar().weights(geom)
        assert w.shape == (8, 8)
        np.testing.assert_array_equal(w, 1.0)


class TestSpatialHann:
    def test_symmetric_zero_at_edges(self, geom: SpatialRectangular) -> None:
        w = SpatialHann().weights(geom)
        assert w.shape == (8, 8)
        np.testing.assert_allclose(w[0, :], 0.0, atol=1e-12)
        np.testing.assert_allclose(w[-1, :], 0.0, atol=1e-12)
        # Interior is strictly positive
        assert (w[2:-2, 2:-2] > 0).all()


class TestSpatialTukey:
    def test_alpha_zero_is_boxcar(self, geom: SpatialRectangular) -> None:
        w = SpatialTukey(alpha=0.0).weights(geom)
        np.testing.assert_allclose(w, 1.0)


class TestSpatialGaussian:
    def test_peak_centre(self, geom: SpatialRectangular) -> None:
        w = SpatialGaussian(sigma=0.5).weights(geom)
        ctr = w[w.shape[0] // 2, w.shape[1] // 2]
        edge = w[0, 0]
        assert ctr > edge


class TestSpatialCustom:
    def test_calls_user_fn(self, geom: SpatialRectangular) -> None:
        called: list[bool] = []

        def fn(g):
            called.append(True)
            return np.full(g.size, 0.5)

        w = SpatialCustom(fn=fn).weights(geom)
        assert called == [True]
        np.testing.assert_array_equal(w, 0.5)
