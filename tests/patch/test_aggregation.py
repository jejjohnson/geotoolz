"""Tests for the spatial `Aggregation` family."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor
from rasterio.windows import Window

from geotoolz.patch import (
    Patch,
    SpatialByIndex,
    SpatialHardVote,
    SpatialInvVarWeightedMean,
    SpatialMean,
    SpatialMedian,
    SpatialOverlapAdd,
    SpatialSum,
    SpatialVariance,
)


def _patch(data, row: int, col: int, weights=None, *, shape=None) -> Patch:
    if shape is None:
        shape = data.shape[-2:]  # type: ignore[union-attr]
    h, w = shape
    win = Window(col_off=col, row_off=row, width=w, height=h)
    return Patch(data=data, anchor=(row, col), indices=win, weights=weights)


@pytest.fixture
def empty_field() -> GeoTensor:
    return GeoTensor(
        values=np.zeros((4, 4), dtype=np.float32),
        transform=rasterio.Affine.identity(),
        crs="EPSG:32630",
    )


class TestSpatialSum:
    def test_disjoint_patches(self, empty_field: GeoTensor) -> None:
        p1 = _patch(np.ones((2, 2)), 0, 0)
        p2 = _patch(np.full((2, 2), 3.0), 2, 2)
        out = SpatialSum().merge([p1, p2], empty_field)
        assert out.sum() == 4 * 1 + 4 * 3
        assert out[0, 0] == 1.0
        assert out[3, 3] == 3.0


class TestSpatialMean:
    def test_overlap_mean(self, empty_field: GeoTensor) -> None:
        p1 = _patch(np.full((2, 2), 2.0), 0, 0)
        p2 = _patch(np.full((2, 2), 6.0), 1, 1)  # overlaps p1 at (1,1)
        out = SpatialMean().merge([p1, p2], empty_field)
        assert out[1, 1] == pytest.approx(4.0)


class TestSpatialVariance:
    def test_variance_zero_for_constant_patches(self, empty_field: GeoTensor) -> None:
        p1 = _patch(np.full((2, 2), 5.0), 0, 0)
        p2 = _patch(np.full((2, 2), 5.0), 0, 0)
        out = SpatialVariance().merge([p1, p2], empty_field)
        assert out[0, 0] == pytest.approx(0.0)


class TestSpatialOverlapAdd:
    def test_uniform_weights_reconstruct(self, empty_field: GeoTensor) -> None:
        # Two patches, no overlap, boxcar weights -> exact reconstruction
        p1 = _patch(np.ones((2, 2)), 0, 0, weights=np.ones((2, 2)))
        p2 = _patch(np.full((2, 2), 5.0), 2, 2, weights=np.ones((2, 2)))
        out = SpatialOverlapAdd().merge([p1, p2], empty_field)
        assert out[0, 0] == 1.0
        assert out[3, 3] == 5.0

    def test_multiband_raster_trailing_axis(self) -> None:
        """The (row, col) slicer must target the trailing two axes when the
        domain has a leading band/time dim. Regression for the bug where
        ``acc[(row_slice, col_slice)]`` indexed the first two axes instead.
        """
        domain = GeoTensor(
            values=np.zeros((3, 4, 4), dtype=np.float32),  # (band, H, W)
            transform=rasterio.Affine.identity(),
            crs="EPSG:32630",
        )
        # Two non-overlapping (3-band, 2x2) patches at (0,0) and (2,2).
        p1 = _patch(np.full((3, 2, 2), 1.0), 0, 0, weights=np.ones((2, 2)))
        p2 = _patch(np.full((3, 2, 2), 5.0), 2, 2, weights=np.ones((2, 2)))
        out = SpatialOverlapAdd().merge([p1, p2], domain)
        assert out.shape == (3, 4, 4)
        np.testing.assert_array_equal(out[:, :2, :2], 1.0)
        np.testing.assert_array_equal(out[:, 2:, 2:], 5.0)

    def test_normalisation_with_overlap(self, empty_field: GeoTensor) -> None:
        # Two identical patches with non-uniform weights at the same location.
        # OverlapAdd normalises (sum w*x / sum w) → the per-cell average, which
        # equals x because both patches have the same data.
        w = np.array([[0.5, 1.0], [1.0, 0.5]])
        p1 = _patch(np.full((2, 2), 3.0), 0, 0, weights=w)
        p2 = _patch(np.full((2, 2), 3.0), 0, 0, weights=w)
        out = SpatialOverlapAdd().merge([p1, p2], empty_field)
        np.testing.assert_allclose(out[:2, :2], 3.0)


class TestSpatialInvVarWeightedMean:
    def test_two_patches_gaussian_merge(self, empty_field: GeoTensor) -> None:
        # Two overlapping Gaussian patches with the same mean -> global mean
        # equals it; global variance equals 1 / (1/var1 + 1/var2).
        mu = np.full((2, 2), 3.0)
        var1 = np.full((2, 2), 4.0)
        var2 = np.full((2, 2), 1.0)
        p1 = _patch((mu, var1), 0, 0, shape=(2, 2))
        p2 = _patch((mu, var2), 0, 0, shape=(2, 2))
        out = SpatialInvVarWeightedMean().merge([p1, p2], empty_field)
        np.testing.assert_allclose(out["mu"][0, 0], 3.0)
        np.testing.assert_allclose(out["var"][0, 0], 1.0 / (1 / 4 + 1 / 1))


class TestSpatialByIndex:
    def test_returns_anchor_keyed_dict(self, empty_field: GeoTensor) -> None:
        p1 = _patch(np.array([[1.0]]), 0, 0)
        p2 = _patch(np.array([[2.0]]), 1, 1)
        out = SpatialByIndex().merge([p1, p2], empty_field)
        assert (0, 0) in out
        assert (1, 1) in out


class TestSpatialHardVote:
    def test_majority(self, empty_field: GeoTensor) -> None:
        # Three patches at (0,0); 2 vote for class 1, 1 votes for class 0
        p1 = _patch(np.zeros((2, 2), dtype=int), 0, 0)
        p2 = _patch(np.ones((2, 2), dtype=int), 0, 0)
        p3 = _patch(np.ones((2, 2), dtype=int), 0, 0)
        out = SpatialHardVote(n_classes=2).merge([p1, p2, p3], empty_field)
        assert (out[:2, :2] == 1).all()


class TestSpatialMedian:
    def test_warns_when_streaming(self, empty_field: GeoTensor) -> None:
        # SpatialMedian.streaming_safe == False — `_warn_if_unsafe_streaming`
        # is called from SpatialPatcher.merge, not from Median.merge itself.
        from geotoolz.patch._src.spatial.aggregation import (
            _warn_if_unsafe_streaming,
        )

        with pytest.warns(RuntimeWarning, match="streaming_safe = False"):
            _warn_if_unsafe_streaming(SpatialMedian())

    def test_median_value(self, empty_field: GeoTensor) -> None:
        p1 = _patch(np.full((2, 2), 1.0), 0, 0)
        p2 = _patch(np.full((2, 2), 5.0), 0, 0)
        p3 = _patch(np.full((2, 2), 3.0), 0, 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = SpatialMedian().merge([p1, p2, p3], empty_field)
        assert out[0, 0] == 3.0
