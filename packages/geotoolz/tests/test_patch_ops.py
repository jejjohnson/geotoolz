"""Tests for the operator wrappers (`GridSampler`, `ApplyToChips`, `Stitch`).

The wrappers re-export `geopatcher` primitives at runtime; skip cleanly
when the optional ``[patch]`` extra (which pulls in geopatcher) isn't
installed.
"""

from __future__ import annotations

import pytest


pytest.importorskip(
    "geopatcher",
    reason="geotoolz.patch_ops bridge requires the [patch] extra (geopatcher)",
)

import numpy as np
import rasterio
from geopatcher import (
    Patch,
    RasterField,
    SpatialBoxcar,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRectangular,
    SpatialRegularStride,
)
from georeader.geotensor import GeoTensor
from pipekit import Lambda

from geotoolz import Sequential
from geotoolz.geom._src import array as geom_array
from geotoolz.patch_ops import (
    ApplyToChips,
    GridSampler,
    SpatialTriangular,
    Stitch,
)


@pytest.fixture
def field() -> RasterField:
    # 2-D field so OverlapAdd's row/col slicer matches the domain shape.
    arr = np.ones((16, 16), dtype=np.float32)
    gt = GeoTensor(
        values=arr,
        transform=rasterio.Affine.identity(),
        crs="EPSG:32630",
    )
    return RasterField(gt)


@pytest.fixture
def patcher() -> SpatialPatcher:
    return SpatialPatcher(
        geometry=SpatialRectangular(size=(8, 8)),
        sampler=SpatialRegularStride(step=8),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )


class TestGridSampler:
    def test_returns_list_of_patches(
        self, field: RasterField, patcher: SpatialPatcher
    ) -> None:
        op = GridSampler(patcher)
        patches = op(field)
        assert isinstance(patches, list)
        assert all(isinstance(p, Patch) for p in patches)
        assert len(patches) == 4  # 2x2 tiles


def test_spatial_triangular_matches_geom_feather_kernel() -> None:
    weights = SpatialTriangular(width=2).weights(SpatialRectangular(size=(5, 7)))
    np.testing.assert_array_equal(weights, geom_array.feather_weights((5, 7), width=2))
    assert weights[0, 0] == pytest.approx(0.25)
    assert weights[2, 3] == pytest.approx(1.0)
    np.testing.assert_array_equal(weights, np.flip(weights, axis=0))
    np.testing.assert_array_equal(weights, np.flip(weights, axis=1))

    small = SpatialTriangular(width=2).weights(SpatialRectangular(size=(3, 3)))
    np.testing.assert_array_equal(
        small,
        np.array(
            [
                [0.25, 0.5, 0.25],
                [0.5, 1.0, 0.5],
                [0.25, 0.5, 0.25],
            ],
            dtype=np.float32,
        ),
    )


class TestApplyToChips:
    def test_each_chip_runs_through_operator(
        self, field: RasterField, patcher: SpatialPatcher
    ) -> None:
        patches = list(patcher.split(field))
        double = Lambda(lambda gt: np.asarray(gt) * 2.0, name="double")
        out = ApplyToChips(double)(patches)
        assert len(out) == len(patches)
        for src, dst in zip(patches, out, strict=True):
            assert dst.anchor == src.anchor
            np.testing.assert_allclose(dst.data, 2.0)


class TestStitchInSequential:
    def test_chip_predict_stitch_roundtrip(
        self, field: RasterField, patcher: SpatialPatcher
    ) -> None:
        double = Lambda(lambda gt: np.asarray(gt) * 2.0, name="double")
        pipe = Sequential(
            [
                GridSampler(patcher),
                ApplyToChips(double),
                Stitch(SpatialOverlapAdd(), domain=field.reader),
            ]
        )
        result = pipe(field)
        np.testing.assert_allclose(result, 2.0)


class TestLabelSamplers:
    """`StratifiedSample` and `BalancedSampler` — label-aware chip draws."""

    @pytest.fixture
    def labels(self) -> np.ndarray:
        # Three vertical class bands: 0 | 1 | 2, each 10 px wide.
        lab = np.zeros((30, 30), dtype=np.int64)
        lab[:, 10:20] = 1
        lab[:, 20:] = 2
        return lab

    @pytest.fixture
    def scene(self) -> GeoTensor:
        return GeoTensor(
            values=np.arange(30 * 30, dtype=np.float32).reshape(30, 30),
            transform=rasterio.Affine.identity(),
            crs="EPSG:32630",
        )

    @staticmethod
    def _classes(patches: list[Patch], labels: np.ndarray, size: tuple[int, int]):
        ph, pw = size
        return [
            int(labels[r + ph // 2, c + pw // 2])
            for r, c in (p.anchor for p in patches)
        ]

    def test_stratified_largest_remainder_allocation(
        self, labels: np.ndarray, scene: GeoTensor
    ) -> None:
        from collections import Counter

        from geotoolz.patch_ops import StratifiedSample

        op = StratifiedSample(
            labels=labels,
            target_proportions={0: 0.5, 1: 0.3, 2: 0.2},
            n_samples=7,
            size=(4, 4),
            seed=0,
        )
        patches = op(scene)
        assert len(patches) == 7
        # Quotas 3.5 / 2.1 / 1.4 -> base 3/2/1, largest remainder -> class 0.
        assert Counter(self._classes(patches, labels, (4, 4))) == {0: 4, 1: 2, 2: 1}

    def test_stratified_seed_reproducible(
        self, labels: np.ndarray, scene: GeoTensor
    ) -> None:
        from geotoolz.patch_ops import StratifiedSample

        kwargs = dict(
            labels=labels,
            target_proportions={0: 0.5, 1: 0.5},
            n_samples=6,
            size=(4, 4),
            seed=123,
        )
        a = StratifiedSample(**kwargs)(scene)
        b = StratifiedSample(**kwargs)(scene)
        assert [p.anchor for p in a] == [p.anchor for p in b]

    def test_stratified_proportions_must_sum_to_one(self, labels: np.ndarray) -> None:
        from geotoolz.patch_ops import StratifiedSample

        with pytest.raises(ValueError, match="sum to 1"):
            StratifiedSample(
                labels=labels,
                target_proportions={0: 0.5, 1: 0.2},
                n_samples=4,
                size=(4, 4),
            )

    def test_stratified_sparse_class_warns_and_shrinks(self, scene: GeoTensor) -> None:
        from geotoolz.patch_ops import StratifiedSample

        # Class 1 exists at a single centre position only.
        labels = np.zeros((30, 30), dtype=np.int64)
        labels[15, 15] = 1
        op = StratifiedSample(
            labels=labels,
            target_proportions={0: 0.5, 1: 0.5},
            n_samples=8,
            size=(4, 4),
            seed=0,
        )
        with pytest.warns(UserWarning, match="class 1 has only 1"):
            patches = op(scene)
        assert len(patches) == 5  # 4 of class 0 + the single class-1 chip

    def test_chips_carry_shifted_transform(
        self, labels: np.ndarray, scene: GeoTensor
    ) -> None:
        from geotoolz.patch_ops import StratifiedSample

        op = StratifiedSample(
            labels=labels,
            target_proportions={2: 1.0},
            n_samples=3,
            size=(6, 6),
            seed=1,
        )
        for p in op(scene):
            r, c = p.anchor
            assert p.data.shape[-2:] == (6, 6)
            assert p.data.transform.c == pytest.approx(float(c))
            assert p.data.transform.f == pytest.approx(float(r))
            assert p.indices.col_off == c and p.indices.row_off == r
            np.testing.assert_array_equal(
                np.asarray(p.data), np.asarray(scene)[r : r + 6, c : c + 6]
            )

    def test_stratified_composes_with_apply_to_chips(
        self, labels: np.ndarray, scene: GeoTensor
    ) -> None:
        from geotoolz.patch_ops import StratifiedSample

        double = Lambda(lambda gt: np.asarray(gt) * 2.0, name="double")
        pipe = Sequential(
            [
                StratifiedSample(
                    labels=labels,
                    target_proportions={0: 1.0},
                    n_samples=2,
                    size=(4, 4),
                    seed=0,
                ),
                ApplyToChips(double),
            ]
        )
        out = pipe(scene)
        assert len(out) == 2
        for p in out:
            r, c = p.anchor
            np.testing.assert_allclose(
                p.data, np.asarray(scene)[r : r + 4, c : c + 4] * 2.0
            )

    def test_balanced_exact_n_per_class(
        self, labels: np.ndarray, scene: GeoTensor
    ) -> None:
        from collections import Counter

        from geotoolz.patch_ops import BalancedSampler

        op = BalancedSampler(labels=labels, n_per_class=5, size=(4, 4), seed=0)
        patches = op(scene)
        assert Counter(self._classes(patches, labels, (4, 4))) == {0: 5, 1: 5, 2: 5}

    def test_balanced_explicit_classes(
        self, labels: np.ndarray, scene: GeoTensor
    ) -> None:
        from geotoolz.patch_ops import BalancedSampler

        op = BalancedSampler(
            labels=labels, n_per_class=3, size=(4, 4), classes=[0, 2], seed=0
        )
        patches = op(scene)
        classes = set(self._classes(patches, labels, (4, 4)))
        assert classes == {0, 2}
        assert len(patches) == 6

    def test_balanced_sparse_class_warns(self, scene: GeoTensor) -> None:
        from geotoolz.patch_ops import BalancedSampler

        labels = np.zeros((30, 30), dtype=np.int64)
        labels[15, 15] = 1
        op = BalancedSampler(labels=labels, n_per_class=4, size=(4, 4), seed=0)
        with pytest.warns(UserWarning, match="class 1 has only 1"):
            patches = op(scene)
        assert len(patches) == 5

    def test_plain_ndarray_scene(self, labels: np.ndarray) -> None:
        from geotoolz.patch_ops import BalancedSampler

        scene = np.random.default_rng(0).random((3, 30, 30))
        op = BalancedSampler(labels=labels, n_per_class=2, size=(8, 8), seed=0)
        patches = op(scene)
        assert len(patches) == 6
        for p in patches:
            assert isinstance(p.data, np.ndarray)
            assert p.data.shape == (3, 8, 8)

    def test_patch_size_larger_than_labels_raises(self, scene: GeoTensor) -> None:
        from geotoolz.patch_ops import BalancedSampler

        op = BalancedSampler(
            labels=np.zeros((6, 6), dtype=np.int64), n_per_class=1, size=(8, 8)
        )
        with pytest.raises(ValueError, match="exceeds"):
            op(scene)
